import gc
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mobilenetv3_torch import _make_divisible, load_pretrained_from_timm, mobilenet_v3_large
from model.utils import _cxcywh_to_xyxy, _xyxy_to_cxcywh, DefaultBoxGenerator, box_iou, generalized_iou, nms


class ConvBNReLU6(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # BatchNorm2d fails in training mode when N*H*W == 1.
        # Fallback to running-stat normalization for this degenerate case.
        if self.training and x.size(0) * x.size(2) * x.size(3) == 1:
            x = F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=0.0,
                eps=self.bn.eps,
            )
        else:
            x = self.bn(x)
        return self.act(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU6(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
        self.pointwise = ConvBNReLU6(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SSDLiteHeadLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU6(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_heads = nn.ModuleList(
            [SSDLiteHeadLayer(ch, anchors * num_classes) for ch, anchors in zip(in_channels, num_anchors)]
        )
        self.reg_heads = nn.ModuleList(
            [SSDLiteHeadLayer(ch, anchors * 4) for ch, anchors in zip(in_channels, num_anchors)]
        )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_logits: List[torch.Tensor] = []
        bbox_regs: List[torch.Tensor] = []

        for feat, cls_head, reg_head in zip(features, self.cls_heads, self.reg_heads):
            n = feat.size(0)

            cls = cls_head(feat)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(n, -1, self.num_classes)
            cls_logits.append(cls)

            reg = reg_head(feat)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            bbox_regs.append(reg)

        return {
            "cls_logits": torch.cat(cls_logits, dim=1),
            "bbox_regression": torch.cat(bbox_regs, dim=1),
        }


class MobileNetV3SSDLiteBackbone(nn.Module):
    """MobileNetV3-Large backbone adapted for SSDLite feature extraction."""

    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = float(width_mult)

        # Build local MobileNetV3 classification model and reuse stem/features.
        self.backbone_model = mobilenet_v3_large(num_classes=1000, width_mult=self.width_mult)

        # Feature map selection points for 320x320 input:
        # - idx 6  -> 20x20
        # - idx 15 -> 10x10
        self.low_level_index = 6
        self.high_level_index = 15

        c_low = _make_divisible(80 * self.width_mult, 8)
        c_high = _make_divisible(960 * self.width_mult, 8)

        self.proj_low = ConvBNReLU6(c_low, 256, kernel_size=1)
        self.proj_high = ConvBNReLU6(c_high, 512, kernel_size=1)

        self.extra_1 = DepthwiseSeparableConv(512, 512, stride=2)  # 10 -> 5
        self.extra_2 = DepthwiseSeparableConv(512, 256, stride=2)  # 5 -> 3
        self.extra_3 = DepthwiseSeparableConv(256, 256, stride=2)  # 3 -> 2
        self.extra_4 = DepthwiseSeparableConv(256, 128, stride=2)  # 2 -> 1

        self.out_channels: List[int] = [256, 512, 512, 256, 256, 128]

    def load_imagenet_pretrained(self, model_name: str) -> None:
        # Weight transfer is handled by module-order + shape matching.
        load_pretrained_from_timm(self.backbone_model, model_name, verbose=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone_model.stem(x)

        low_level: Optional[torch.Tensor] = None
        high_level: Optional[torch.Tensor] = None

        for idx, block in enumerate(self.backbone_model.features):
            x = block(x)
            if idx == self.low_level_index:
                low_level = x
            if idx == self.high_level_index:
                high_level = x

        if low_level is None or high_level is None:
            raise RuntimeError("Failed to extract MobileNetV3 SSD feature maps.")

        p3 = self.proj_low(low_level)
        p4 = self.proj_high(high_level)
        p5 = self.extra_1(p4)
        p6 = self.extra_2(p5)
        p7 = self.extra_3(p6)
        p8 = self.extra_4(p7)

        return [p3, p4, p5, p6, p7, p8]


class SSDMobile(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aspect_ratios: List[List[float]],
        img_size: int = 320,
        s_min: float = 0.07,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        pretrained_backbone: bool = True,
        pretrained_backbone_model_name: str = "mobilenetv3_large_100",
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.variances = (0.1, 0.2)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.aspect_ratios = aspect_ratios
        self.s_min = s_min
        self.s_max = s_max

        self.pretrained_backbone = pretrained_backbone
        self.pretrained_backbone_model_name = str(pretrained_backbone_model_name).strip() or "mobilenetv3_large_100"
        self.backbone_has_weights_loaded = False

        self.backbone = MobileNetV3SSDLiteBackbone(width_mult=width_mult)
        self.feature_channels = list(self.backbone.out_channels)

        self.anchor_generator = DefaultBoxGenerator(self.aspect_ratios, s_min=self.s_min, s_max=self.s_max)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDLiteHead(self.feature_channels, self.num_anchors, num_classes)

    def load_pretrained_weights(self, device: torch.device):
        if not self.pretrained_backbone:
            print("Pretrained backbone disabled. Skipping weight loading.")
            return

        if self.backbone_has_weights_loaded:
            print("Pretrained weights already loaded for SSDMobile backbone.")
            return

        model_name = self.pretrained_backbone_model_name
        print(
            f"Loading MobileNetV3 backbone ImageNet-1k pretrained weights "
            f"from timm model '{model_name}' to device: {device}"
        )
        try:
            with torch.no_grad():
                self.backbone.load_imagenet_pretrained(model_name)
            self.backbone_has_weights_loaded = True
            print("Successfully loaded MobileNetV3 backbone from timm ImageNet-1k weights.")
        except ImportError:
            print("Failed to import timm. Install with: pip install timm")
            self.backbone_has_weights_loaded = False
        except Exception as e:
            print(f"Failed to load MobileNetV3 timm ImageNet-1k weights: {e}")
            self.backbone_has_weights_loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        if isinstance(features, OrderedDict):
            return list(features.values())
        if isinstance(features, dict):
            return list(features.values())
        if isinstance(features, (list, tuple)):
            return list(features)
        raise TypeError(f"Unexpected backbone output type: {type(features)}")

    def generate_priors(self, feats: List[torch.Tensor], images: torch.Tensor) -> torch.Tensor:
        feat_sizes = [(int(f.shape[-2]), int(f.shape[-1])) for f in feats]
        return self.anchor_generator.generate(feat_sizes, self.img_size, images.device)

    def forward_logits(self, images: List[torch.Tensor] | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images, dim=0)
        feats = self._features(images)
        head_outputs = self.head(feats)
        cls_logits = head_outputs["cls_logits"]
        box_reg = head_outputs["bbox_regression"]
        return cls_logits, box_reg, feats

    def forward(self, images: List[torch.Tensor] | torch.Tensor, targets=None):
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images, dim=0)
        cls_logits, box_reg, feats = self.forward_logits(images)
        priors = self.generate_priors(feats, images)
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

    def multibox_loss(self, cls_logits: torch.Tensor, box_reg: torch.Tensor, targets: List[dict[str, torch.Tensor]], priors: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = cls_logits.device

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

            matched_gt = gt_boxes[best_gt_idx[positives]]
            loc_t = self._encode(matched_gt, priors[positives])
            loc_p = box_reg[b][positives]
            pred_boxes = self._decode(loc_p, priors[positives])
            giou = generalized_iou(pred_boxes, matched_gt)
            loc_loss += (1.0 - giou).sum()

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

    def post_process(
        self,
        cls_logits: torch.Tensor,
        box_reg: torch.Tensor,
        priors: torch.Tensor,
        img_size=None,
        score_thresh=None,
        pre_nms_topk: Optional[int] = None,
        max_detections: Optional[int] = None,
    ):
        from model.utils import batched_nms
        
        if score_thresh is None:
            score_thresh = self.score_thresh
        if pre_nms_topk is None:
            pre_nms_topk = 400
        if max_detections is None:
            max_detections = 100

        device = cls_logits.device
        num_classes = self.num_classes
        
        # 1. Convert logits to probabilities
        probs = F.softmax(cls_logits, dim=-1) # (B, N, C)
        
        outputs = []
        for b in range(cls_logits.size(0)):
            # 2. Decode all boxes for this batch element
            boxes = self._decode(box_reg[b], priors) # (N, 4)
            scores = probs[b] # (N, C)
            
            # 3. Filter boxes by score threshold (ignoring background class 0)
            # We only consider the best class for each anchor to keep it simple and efficient
            conf, labels = scores[:, 1:].max(dim=1)
            labels = labels + 1 # Adjust back to [1, num_classes-1]
            
            keep = conf > score_thresh
            if not keep.any():
                outputs.append(torch.zeros((0, 6), device=device))
                continue
            
            boxes = boxes[keep]
            conf = conf[keep]
            labels = labels[keep]
            
            # 4. Limit to pre_nms_topk
            if pre_nms_topk > 0 and boxes.size(0) > pre_nms_topk:
                topk_idx = torch.topk(conf, k=pre_nms_topk).indices
                boxes = boxes[topk_idx]
                conf = conf[topk_idx]
                labels = labels[topk_idx]
            
            # 5. Apply Batched NMS (class-aware NMS)
            keep_idx = batched_nms(boxes, conf, labels, self.nms_thresh)
            
            boxes = boxes[keep_idx]
            conf = conf[keep_idx]
            labels = labels[keep_idx]
            
            # 6. Combine into final detections [x1, y1, x2, y2, score, label]
            preds = torch.cat([boxes, conf.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
            
            # 7. Final limit to max_detections
            if max_detections > 0 and preds.size(0) > max_detections:
                top_idx = torch.topk(preds[:, 4], k=max_detections).indices
                preds = preds[top_idx]
                
            outputs.append(preds)

        return outputs
