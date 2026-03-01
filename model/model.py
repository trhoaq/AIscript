import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from model.ghostnet import GhostNet, _make_divisible
from model.utils import DefaultBoxGenerator, _xyxy_to_cxcywh, _cxcywh_to_xyxy
from typing import List, Tuple

# --- Model Components ---

class PConv(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class GhostNetBackbone(nn.Module):
    def __init__(self, width=0.5, timm_pretrained=True):
        super().__init__()
        # GhostNet Configuration
        cfgs = [
            [[3,  16,  16, 0, 1]],
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            [[5,  72,  40, 0.25, 2]], # C3 feature
            [[5, 120,  40, 0.25, 1]],
            [[3, 240,  80, 0, 2]],     # C4 feature
            [[3, 200,  80, 0, 1], [3, 184,  80, 0, 1], [3, 184,  80, 0, 1], [3, 480, 112, 0.25, 1], [3, 672, 112, 0.25, 1]],
            [[5, 672, 160, 0.25, 2]],    # C5 feature
            [[5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1], [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1]]
        ]
        model = GhostNet(cfgs, width=width)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks
        self.out_channels = [_make_divisible(40*width, 4), _make_divisible(112*width, 4), _make_divisible(160*width, 4)]
        
        # Indices for feature extraction
        self.feature_indices = [4, 6, 8] 
        
        if timm_pretrained:
            self._load_timm_pretrained(width)

    def _load_timm_pretrained(self, width):
        if width != 0.5:
            print("GhostNetBackbone: timm pretrained only configured for width=0.5, skipping.")
            return
        try:
            import timm  # type: ignore
        except Exception as e:
            print(f"GhostNetBackbone: timm not available ({e}), skipping pretrained load.")
            return
        try:
            timm_model = timm.create_model("ghostnet_050", pretrained=True)
            missing, unexpected = self.load_state_dict(timm_model.state_dict(), strict=False)
            if missing or unexpected:
                print(f"GhostNetBackbone: loaded timm weights with missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"GhostNetBackbone: failed to load timm ghostnet_050 pretrained weights ({e}).")

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feature_indices:
                features.append(x)
        return features

class FPNLitePConv(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        self.pconvs = nn.ModuleList([PConv(out_channels) for _ in in_channels_list])
        self.extra_p6 = DepthwiseSeparableConv(out_channels, out_channels, stride=2)
        self.extra_p7 = DepthwiseSeparableConv(out_channels, out_channels, stride=2)

    def forward(self, inputs):
        laterals = [lat(inputs[i]) for i, lat in enumerate(self.lateral_convs)]
        p5 = laterals[2]
        p4 = laterals[1] + F.interpolate(p5, size=laterals[1].shape[-2:], mode="nearest")
        p3 = laterals[0] + F.interpolate(p4, size=laterals[0].shape[-2:], mode="nearest")
        p3, p4, p5 = self.pconvs[0](p3), self.pconvs[1](p4), self.pconvs[2](p5)
        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(p6)
        return [p3, p4, p5, p6, p7]

# --- Main Model ---

class SSDGhost(nn.Module):
    def __init__(self, num_classes, width=0.5, img_size=256, score_thresh=0.05, nms_thresh=0.45, backbone_pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.variances = (0.1, 0.2)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        
        # Aspect ratios for P3, P4, P5, P6, P7 feature maps
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2]]
        self.anchor_generator = DefaultBoxGenerator(self.aspect_ratios)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()

        self.backbone = GhostNetBackbone(width=width, timm_pretrained=backbone_pretrained)
        self.fpn = FPNLitePConv(self.backbone.out_channels, out_channels=64)
        
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        fpn_out_channels = 64
        for n in self.num_anchors:
            self.cls_heads.append(nn.Sequential(nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, 1, 1, groups=fpn_out_channels, bias=False), nn.Conv2d(fpn_out_channels, n * num_classes, 1)))
            self.reg_heads.append(nn.Sequential(nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, 1, 1, groups=fpn_out_channels, bias=False), nn.Conv2d(fpn_out_channels, n * 4, 1)))

    def _features(self, x):
        return self.fpn(self.backbone(x))

    def forward_logits(self, x):
        # Input can be a list of images (training) or a tensor (inference)
        if isinstance(x, list):
            x = torch.stack(x, dim=0)

        feats = self._features(x)
        cls_logits, bbox_regs = [], []
        for i, feat in enumerate(feats):
            cls = self.cls_heads[i](feat).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            reg = self.reg_heads[i](feat).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            cls_logits.append(cls)
            bbox_regs.append(reg)

        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_regs = torch.cat(bbox_regs, dim=1)
        return cls_logits, bbox_regs, feats

    def forward(self, x, targets=None):
        cls_logits, bbox_regs, feats = self.forward_logits(x)
        feat_sizes = [(f.size(2), f.size(3)) for f in feats]
        priors = self.anchor_generator.generate(feat_sizes, self.img_size, cls_logits.device)

        if self.training and targets is not None:
            return self.multibox_loss(cls_logits, bbox_regs, targets, priors)

        return self.predict(cls_logits, bbox_regs, priors)

    def _encode(self, gt: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        gt_c = _xyxy_to_cxcywh(gt)
        pr_c = _xyxy_to_cxcywh(priors)
        loc = torch.zeros_like(gt_c)
        loc[:, :2] = (gt_c[:, :2] - pr_c[:, :2]) / (pr_c[:, 2:] * self.variances[0])
        loc[:, 2:] = torch.log(gt_c[:, 2:] / pr_c[:, 2:]) / self.variances[1]
        return loc

    def _decode(self, loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        pr_c = _xyxy_to_cxcywh(priors)
        boxes = torch.zeros_like(loc)
        boxes[:, :2] = loc[:, :2] * self.variances[0] * pr_c[:, 2:] + pr_c[:, :2]
        boxes[:, 2:] = torch.exp(loc[:, 2:] * self.variances[1]) * pr_c[:, 2:]
        return _cxcywh_to_xyxy(boxes)

    def multibox_loss(self, cls_p: torch.Tensor, reg_p: torch.Tensor, targets: List[Dict[str, torch.Tensor]], priors: torch.Tensor):
        device = cls_p.device
        
        # Defensive device placement for all inputs
        reg_p = reg_p.to(device)
        priors = priors.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        batch_size = cls_p.size(0)
        loc_loss, cls_loss, total_pos = 0.0, 0.0, 0
        
        for b in range(batch_size):
            gt_boxes, gt_labels = targets[b]["boxes"], targets[b]["labels"]
            if gt_boxes.numel() == 0: continue
            
            ious = box_iou(gt_boxes, priors)
            best_prior_iou, best_prior_idx = ious.max(dim=1)
            best_gt_iou, best_gt_idx = ious.max(dim=0)
            best_gt_iou[best_prior_idx] = 1.0
            best_gt_idx[best_prior_idx] = torch.arange(best_prior_idx.size(0), device=device)
            
            pos = best_gt_iou >= 0.5
            num_pos = pos.sum().item()
            if num_pos == 0: continue
            total_pos += num_pos
            
            loc_t = self._encode(gt_boxes[best_gt_idx[pos]], priors[pos])
            loc_loss += F.smooth_l1_loss(reg_p[b][pos], loc_t, reduction="sum")
            
            cls_t = torch.zeros(priors.size(0), dtype=torch.long, device=device)
            cls_t[pos] = gt_labels[best_gt_idx[pos]]
            conf_loss_all = F.cross_entropy(cls_p[b], cls_t, reduction="none")
            
            neg = ~pos
            num_neg = min(neg.sum().item(), 3 * num_pos)
            neg_loss, _ = conf_loss_all[neg].sort(descending=True)
            
            cls_loss += conf_loss_all[pos].sum() + neg_loss[:num_neg].sum()

        total_pos = max(1, total_pos)
        return {"bbox_regression": loc_loss / total_pos, "classification": cls_loss / total_pos}

    def predict(self, cls_p, reg_p, priors):
        probs = F.softmax(cls_p, dim=-1)
        results = []
        for b in range(cls_p.size(0)):
            boxes = self._decode(reg_p[b], priors)
            scores = probs[b]
            out_boxes, out_scores, out_labels = [], [], []
            for c in range(1, self.num_classes):
                c_scores = scores[:, c]
                keep = c_scores > self.score_thresh
                if keep.sum() == 0: continue
                c_boxes, c_scores = boxes[keep], c_scores[keep]
                keep_idx = nms(c_boxes, c_scores, self.nms_thresh)
                out_boxes.append(c_boxes[keep_idx]); out_scores.append(c_scores[keep_idx])
                out_labels.append(torch.full((keep_idx.numel(),), c, device=cls_p.device, dtype=torch.long))
            if out_boxes:
                results.append({"boxes": torch.cat(out_boxes), "scores": torch.cat(out_scores), "labels": torch.cat(out_labels)})
            else:
                results.append({"boxes": torch.zeros((0, 4), device=cls_p.device), "scores": torch.zeros(0, device=cls_p.device), "labels": torch.zeros(0, dtype=torch.long, device=cls_p.device)})
        return results

