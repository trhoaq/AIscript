from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import box_iou, nms
from model.utils import DefaultBoxGenerator, _xyxy_to_cxcywh, _cxcywh_to_xyxy

class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        for c, a in zip(in_channels, num_anchors):
            self.cls_heads.append(nn.Conv2d(c, a * num_classes, kernel_size=3, padding=1))
            self.box_heads.append(nn.Conv2d(c, a * 4, kernel_size=3, padding=1))

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_out = []
        box_out = []
        for x, cls_h, box_h in zip(feats, self.cls_heads, self.box_heads):
            cls = cls_h(x)
            box = box_h(x)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            box = box.permute(0, 2, 3, 1).contiguous()
            cls_out.append(cls.view(cls.size(0), -1, self.num_classes))
            box_out.append(box.view(box.size(0), -1, 4))
        return torch.cat(cls_out, dim=1), torch.cat(box_out, dim=1)


class SSDMobile(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aspect_ratios: List[List[float]],
        img_size: int = 512,
        s_min: float = 0.07,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        pretrained_backbone: bool = True, # New parameter
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.variances = (0.1, 0.2)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.anchor_generator = DefaultBoxGenerator(aspect_ratios, s_min=s_min, s_max=s_max)

        # Initialize backbone without pretrained weights initially
        # Weights will be loaded manually after the model is moved to the target device, if pretrained_backbone is True
        backbone = mobilenet_v3_large(weights=None) 
        
        # Hard-coded indices for feature extraction from Torchvision's MobileNetV3-Large
        self.backbone = backbone.features
        self.feature_indices = [6, 12, 16] # Indices for C3, C4, C5 layers
        
        self.pretrained_backbone = pretrained_backbone
        self.backbone_has_weights_loaded = False
        if self.pretrained_backbone:
            self._weights_enum = MobileNet_V3_Large_Weights.DEFAULT

        self.fpn_out_channels = 128
        self.lateral_c3 = nn.Conv2d(40, self.fpn_out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(112, self.fpn_out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(960, self.fpn_out_channels, kernel_size=1)
        self.smooth_p3 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1)
        self.p6 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1)
        self.p8 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, stride=2, padding=1)

        self.feature_channels = [self.fpn_out_channels] * 6
        self.num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(self.feature_channels, self.num_anchors, num_classes)

    def load_pretrained_weights(self, device: torch.device):
        if not self.pretrained_backbone:
            print("Pretrained backbone disabled. Skipping weight loading.")
            return

        if self.backbone_has_weights_loaded:
            print("Pretrained weights already loaded for SSDMobile backbone.")
            return

        print(f"Loading MobileNetV3 pretrained weights to device: {device}")
        try:
            # Download weights on CPU first
            state_dict = self._weights_enum.get_state_dict(progress=True)
            self.backbone.load_state_dict(state_dict)
            self.backbone_has_weights_loaded = True
            print("Successfully loaded MobileNetV3 pretrained weights.")
        except Exception as e:
            print(f"Failed to load MobileNetV3 pretrained weights: {e}")
            self.backbone_has_weights_loaded = False

    def _features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        c3_idx, c4_idx, c5_idx = self.feature_indices
        
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == c3_idx:
                features.append(x)
            elif i == c4_idx:
                features.append(x)
            elif i == c5_idx:
                features.append(x)
        
        if len(features) != 3:
            raise RuntimeError("Backbone feature extraction failed. Check backbone indices.")
        
        c3, c4, c5 = features

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        p8 = self.p8(F.relu(p7))

        return [p3, p4, p5, p6, p7, p8]

    def forward_logits(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images, dim=0)
        feats = self._features(images)
        cls_logits, box_reg = self.head(feats)
        return cls_logits, box_reg, feats

    def forward(self, images: List[torch.Tensor], targets=None):
        cls_logits, box_reg, feats = self.forward_logits(images)
        feat_sizes = [(f.size(2), f.size(3)) for f in feats]
        priors = self.anchor_generator.generate(feat_sizes, self.img_size, cls_logits.device)
        if targets is None:
            return self.post_process(cls_logits, box_reg, priors)
        loss_dict = self.multibox_loss(cls_logits, box_reg, targets, priors)
        return loss_dict

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

    def multibox_loss(self, cls_logits: torch.Tensor, box_reg: torch.Tensor, targets: List[dict[str, torch.Tensor]], priors: torch.Tensor) -> dict:
        device = cls_logits.device
        
        # Defensive device placement for all inputs
        box_reg = box_reg.to(device)
        priors = priors.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        batch_size = cls_logits.size(0)

        loc_loss = 0.0
        cls_loss = 0.0
        total_pos = 0
        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]
            if gt_boxes.numel() == 0:
                continue
            ious = box_iou(gt_boxes, priors)
            best_prior_iou, best_prior_idx = ious.max(dim=1)
            best_gt_iou, best_gt_idx = ious.max(dim=0)

            best_gt_iou[best_prior_idx] = 1.0
            best_gt_idx[best_prior_idx] = torch.arange(best_prior_idx.size(0), device=device)

            positives = best_gt_iou >= 0.5
            num_pos = positives.sum().item()
            if num_pos == 0:
                continue
            total_pos += num_pos

            loc_t = self._encode(gt_boxes[best_gt_idx[positives]], priors[positives])
            loc_p = box_reg[b][positives]
            loc_loss += F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

            cls_targets = torch.zeros(priors.size(0), dtype=torch.long, device=device)
            cls_targets[positives] = gt_labels[best_gt_idx[positives]]
            cls_p = cls_logits[b]
            conf_loss_all = F.cross_entropy(cls_p, cls_targets, reduction="none")

            neg_mask = ~positives
            neg_loss = conf_loss_all[neg_mask]
            num_neg_to_mine = min(neg_mask.sum().item(), 3 * num_pos)
            
            cls_loss_pos = conf_loss_all[positives].sum()
            cls_loss_neg, _ = neg_loss.sort(descending=True)
            cls_loss += cls_loss_pos + cls_loss_neg[:num_neg_to_mine].sum()

        total_pos = max(1, total_pos)
        return {
            "bbox_regression": loc_loss / total_pos,
            "classification": cls_loss / total_pos,
        }

    def post_process(self, cls_logits: torch.Tensor, box_reg: torch.Tensor, priors: torch.Tensor, img_size=None, score_thresh=None):
        if score_thresh is None: score_thresh = self.score_thresh
        device = cls_logits.device
        probs = F.softmax(cls_logits, dim=-1)
        outputs = []
        for b in range(cls_logits.size(0)):
            boxes = self._decode(box_reg[b], priors)
            scores = probs[b]
            out_boxes = []
            out_scores = []
            out_labels = []
            for c in range(1, self.num_classes):
                cls_scores = scores[:, c]
                keep = cls_scores > score_thresh
                if keep.sum() == 0:
                    continue
                boxes_c = boxes[keep]
                scores_c = cls_scores[keep]
                keep_idx = nms(boxes_c, scores_c, self.nms_thresh)
                out_boxes.append(boxes_c[keep_idx])
                out_scores.append(scores_c[keep_idx])
                out_labels.append(torch.full((keep_idx.numel(),), c, device=device, dtype=torch.long))
            
            if out_boxes:
                # Combine boxes, scores, and labels into a single tensor [N, 6] (x1, y1, x2, y2, score, label)
                boxes_cat = torch.cat(out_boxes, dim=0)
                scores_cat = torch.cat(out_scores, dim=0).unsqueeze(1)
                labels_cat = torch.cat(out_labels, dim=0).unsqueeze(1).float()
                outputs.append(torch.cat([boxes_cat, scores_cat, labels_cat], dim=1))
            else:
                outputs.append(torch.zeros((0, 6), device=device))
        return outputs
