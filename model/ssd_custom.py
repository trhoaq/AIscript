from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou, nms

from model.utils import _cxcywh_to_xyxy, _xyxy_to_cxcywh


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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.variances = (0.1, 0.2)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        # Keep these fields for API compatibility with existing training scripts.
        self.aspect_ratios = aspect_ratios
        self.s_min = s_min
        self.s_max = s_max

        self.pretrained_backbone = pretrained_backbone
        self.backbone_has_weights_loaded = False
        self._weights_enum = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None

        ssdlite_model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
        self.backbone = ssdlite_model.backbone
        self.head = ssdlite_model.head
        self.anchor_generator = ssdlite_model.anchor_generator
        self.feature_channels = list(getattr(self.backbone, "out_channels", [672, 480, 512, 256, 256, 128]))

    def load_pretrained_weights(self, device: torch.device):
        if not self.pretrained_backbone:
            print("Pretrained backbone disabled. Skipping weight loading.")
            return

        if self.backbone_has_weights_loaded:
            print("Pretrained weights already loaded for SSDMobile backbone.")
            return

        print(f"Loading SSDLite MobileNetV3 backbone weights to device: {device}")
        try:
            source_model = ssdlite320_mobilenet_v3_large(weights=self._weights_enum)
            missing, unexpected = self.backbone.load_state_dict(source_model.backbone.state_dict(), strict=False)
            del source_model
            if missing or unexpected:
                print(f"SSDLite backbone load: missing={len(missing)} unexpected={len(unexpected)}")
            self.backbone_has_weights_loaded = True
            print("Successfully loaded SSDLite MobileNetV3 backbone weights.")
        except Exception as e:
            print(f"Failed to load SSDLite MobileNetV3 backbone weights: {e}")
            self.backbone_has_weights_loaded = False

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
        image_sizes = [(int(images.shape[-2]), int(images.shape[-1]))] * images.shape[0]
        image_list = ImageList(images, image_sizes)
        priors_per_image = self.anchor_generator(image_list, feats)
        return priors_per_image[0]

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
        if score_thresh is None:
            score_thresh = self.score_thresh
        if pre_nms_topk is None:
            pre_nms_topk = 400
        if max_detections is None:
            max_detections = 100

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

                # Cap the candidate set per class to keep NMS tractable.
                if pre_nms_topk > 0 and scores_c.numel() > pre_nms_topk:
                    topk_idx = torch.topk(scores_c, k=pre_nms_topk).indices
                    boxes_c = boxes_c[topk_idx]
                    scores_c = scores_c[topk_idx]

                keep_idx = nms(boxes_c, scores_c, self.nms_thresh)
                out_boxes.append(boxes_c[keep_idx])
                out_scores.append(scores_c[keep_idx])
                out_labels.append(torch.full((keep_idx.numel(),), c, device=device, dtype=torch.long))
            
            if out_boxes:
                # Combine boxes, scores, and labels into a single tensor [N, 6] (x1, y1, x2, y2, score, label)
                boxes_cat = torch.cat(out_boxes, dim=0)
                scores_cat = torch.cat(out_scores, dim=0).unsqueeze(1)
                labels_cat = torch.cat(out_labels, dim=0).unsqueeze(1).float()
                preds = torch.cat([boxes_cat, scores_cat, labels_cat], dim=1)

                # Keep only global top-scoring detections per image.
                if max_detections > 0 and preds.size(0) > max_detections:
                    top_idx = torch.topk(preds[:, 4], k=max_detections).indices
                    preds = preds[top_idx]

                outputs.append(preds)
            else:
                outputs.append(torch.zeros((0, 6), device=device))
        return outputs
