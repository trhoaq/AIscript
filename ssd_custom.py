from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import box_iou, nms


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor / 2) // divisor * divisor)


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


class DefaultBoxGenerator:
    def __init__(
        self,
        aspect_ratios: List[List[float]],
        s_min: float = 0.07,
        s_max: float = 0.95,
    ) -> None:
        self.aspect_ratios = aspect_ratios
        self.s_min = s_min
        self.s_max = s_max

    def num_anchors_per_location(self) -> List[int]:
        counts = []
        for ars in self.aspect_ratios:
            counts.append(2 + 2 * len(ars))
        return counts

    def _scales(self, m: int) -> List[float]:
        if m == 1:
            return [self.s_max]
        return [self.s_min + (self.s_max - self.s_min) * k / (m - 1) for k in range(m)]

    def generate(self, feat_sizes: List[Tuple[int, int]], img_size: int, device: torch.device) -> torch.Tensor:
        m = len(feat_sizes)
        scales = self._scales(m)
        priors: List[torch.Tensor] = []
        for k, (fh, fw) in enumerate(feat_sizes):
            sk = scales[k]
            sk1 = scales[min(k + 1, m - 1)]
            for i in range(fh):
                cy = (i + 0.5) / fh
                for j in range(fw):
                    cx = (j + 0.5) / fw
                    # aspect ratio 1, size sk
                    priors.append(torch.tensor([cx, cy, sk, sk], device=device))
                    # aspect ratio 1, size sqrt(sk*sk1)
                    s_prime = (sk * sk1) ** 0.5
                    priors.append(torch.tensor([cx, cy, s_prime, s_prime], device=device))
                    for ar in self.aspect_ratios[k]:
                        r = ar ** 0.5
                        priors.append(torch.tensor([cx, cy, sk * r, sk / r], device=device))
                        priors.append(torch.tensor([cx, cy, sk / r, sk * r], device=device))
        priors_t = torch.stack(priors, dim=0)
        priors_t = torch.clamp(priors_t, 0.0, 1.0)
        priors_xyxy = _cxcywh_to_xyxy(priors_t) * img_size
        return priors_xyxy


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


class SSD512(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aspect_ratios: List[List[float]],
        img_size: int = 512,
        s_min: float = 0.07,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.variances = (0.1, 0.2)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.anchor_generator = DefaultBoxGenerator(aspect_ratios, s_min=s_min, s_max=s_max)

        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)
        self.backbone = backbone.features
        self.backbone_out = 960

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_out, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            ),
        ])

        self.feature_channels = [self.backbone_out, 512, 256, 256, 128, 128]
        self.num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(self.feature_channels, self.num_anchors, num_classes)

    def _features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.backbone(x)
        feats.append(x)
        for layer in self.extras:
            x = layer(x)
            feats.append(x)
        return feats

    def forward(self, images: List[torch.Tensor], targets=None):
        x = torch.stack(images, dim=0)
        feats = self._features(x)
        cls_logits, box_reg = self.head(feats)
        feat_sizes = [(f.size(2), f.size(3)) for f in feats]
        priors = self.anchor_generator.generate(feat_sizes, self.img_size, cls_logits.device)
        if targets is None:
            return self.predict(cls_logits, box_reg, priors)
        loss_dict = self.multibox_loss(cls_logits, box_reg, targets, priors)
        return loss_dict

    def _encode(self, gt: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        gt_c = _xyxy_to_cxcywh(gt)
        pr_c = _xyxy_to_cxcywh(priors)
        loc = torch.zeros_like(gt_c)
        loc[:, 0] = (gt_c[:, 0] - pr_c[:, 0]) / (pr_c[:, 2] * self.variances[0])
        loc[:, 1] = (gt_c[:, 1] - pr_c[:, 1]) / (pr_c[:, 3] * self.variances[0])
        loc[:, 2] = torch.log(gt_c[:, 2] / pr_c[:, 2]) / self.variances[1]
        loc[:, 3] = torch.log(gt_c[:, 3] / pr_c[:, 3]) / self.variances[1]
        return loc

    def _decode(self, loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        pr_c = _xyxy_to_cxcywh(priors)
        boxes = torch.zeros_like(loc)
        boxes[:, 0] = loc[:, 0] * self.variances[0] * pr_c[:, 2] + pr_c[:, 0]
        boxes[:, 1] = loc[:, 1] * self.variances[0] * pr_c[:, 3] + pr_c[:, 1]
        boxes[:, 2] = torch.exp(loc[:, 2] * self.variances[1]) * pr_c[:, 2]
        boxes[:, 3] = torch.exp(loc[:, 3] * self.variances[1]) * pr_c[:, 3]
        return _cxcywh_to_xyxy(boxes)

    def multibox_loss(self, cls_logits: torch.Tensor, box_reg: torch.Tensor, targets, priors: torch.Tensor) -> dict:
        device = cls_logits.device
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

            matched_boxes = gt_boxes[best_gt_idx]
            matched_labels = gt_labels[best_gt_idx]

            positives = best_gt_iou >= 0.5
            num_pos = positives.sum().item()
            total_pos += num_pos

            loc_t = self._encode(matched_boxes, priors)
            loc_p = box_reg[b]
            if num_pos > 0:
                loc_loss += F.smooth_l1_loss(loc_p[positives], loc_t[positives], reduction="sum")

            cls_targets = torch.zeros(priors.size(0), dtype=torch.long, device=device)
            cls_targets[positives] = matched_labels[positives]
            cls_p = cls_logits[b]
            cls_loss_all = F.cross_entropy(cls_p, cls_targets, reduction="none")

            neg_mask = ~positives
            cls_loss_neg = cls_loss_all[neg_mask]
            num_neg = min(neg_mask.sum().item(), 3 * num_pos if num_pos > 0 else 0)
            if num_neg > 0:
                neg_vals, neg_idx = cls_loss_neg.sort(descending=True)
                neg_keep = neg_idx[:num_neg]
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)[neg_keep]
                cls_loss += cls_loss_all[positives].sum() + cls_loss_all[neg_indices].sum()
            else:
                cls_loss += cls_loss_all[positives].sum()

        total_pos = max(1, total_pos)
        return {
            "bbox_regression": loc_loss / total_pos,
            "classification": cls_loss / total_pos,
        }

    def predict(self, cls_logits: torch.Tensor, box_reg: torch.Tensor, priors: torch.Tensor):
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
                keep = cls_scores > self.score_thresh
                if keep.sum() == 0:
                    continue
                boxes_c = boxes[keep]
                scores_c = cls_scores[keep]
                keep_idx = nms(boxes_c, scores_c, self.nms_thresh)
                out_boxes.append(boxes_c[keep_idx])
                out_scores.append(scores_c[keep_idx])
                out_labels.append(torch.full((keep_idx.numel(),), c, device=device, dtype=torch.long))
            if out_boxes:
                outputs.append({
                    "boxes": torch.cat(out_boxes, dim=0),
                    "scores": torch.cat(out_scores, dim=0),
                    "labels": torch.cat(out_labels, dim=0),
                })
            else:
                outputs.append({"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)})
        return outputs
