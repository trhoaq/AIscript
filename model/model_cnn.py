from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import batched_nms, generalized_iou


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out = out + x
        return F.relu(out, inplace=True)


class SpatialPyramidPooling(nn.Module):
    """SPP layer (stride=1) applied at C5 before FPN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool_specs = (3, 5, 7)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in self.pool_specs])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(self.pools) + 1), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x] + [p(x) for p in self.pools]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class SimpleCNNBackbone(nn.Module):
    """
    Simple CNN backbone with preprocessing stages:
    S1 (stride 2) -> S2 (stride 2) -> C2 (stride 4) -> C3 (stride 8) -> C4 (stride 16) -> C5 (stride 32)
    Single SPP block is applied at C5 before FPN.
    """

    def __init__(self, base_channels: List[int], stem_channels: Optional[List[int]] = None) -> None:
        super().__init__()
        if len(base_channels) != 3:
            raise ValueError("base_channels must have 3 values, e.g., [32, 64, 128].")

        c1, c2, c3 = [int(x) for x in base_channels]

        if stem_channels is None:
            stem_channels = [max(16, c1 // 2), c1]
        if len(stem_channels) != 2:
            raise ValueError("stem_channels must have 2 values, e.g., [16, 32].")
        s1, s2 = [int(x) for x in stem_channels]

        # Preprocessing stages (S1, S2)
        self.stage_s1 = ConvBNReLU(3, s1, kernel_size=3, stride=2)
        self.stage_s2 = ConvBNReLU(s1, s2, kernel_size=3, stride=2)

        # C3 (stride 8)
        self.conv_c3 = ConvBNReLU(s2, c1, kernel_size=3, stride=2)
        self.res_c3 = ResidualBlock(c1, dropout=0.0)

        # C4 (stride 16)
        self.conv_c4 = ConvBNReLU(c1, c2, kernel_size=3, stride=2)
        self.res_c4 = ResidualBlock(c2, dropout=0.0)

        # C5 (stride 32) + single SPP
        self.conv_c5 = ConvBNReLU(c2, c3, kernel_size=3, stride=2)
        self.res_c5 = ResidualBlock(c3, dropout=0.0)
        self.spp_c5 = SpatialPyramidPooling(c3)

        self.out_channels = [s2, c1, c2, c3]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stage_s1(x)
        x = self.stage_s2(x)
        c2 = x

        c3 = self.res_c3(self.conv_c3(x))
        c4 = self.res_c4(self.conv_c4(c3))
        c5 = self.res_c5(self.conv_c5(c4))
        c5 = self.spp_c5(c5)
        return [c2, c3, c4, c5]


class SimpleFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )

        self.extra_p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.extra_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features: [C2, C3, C4, C5]
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(l) for conv, l in zip(self.output_convs, laterals)]
        p6 = self.extra_p6(outs[-1])
        p7 = self.extra_p7(F.relu(p6, inplace=True))
        return outs + [p6, p7]


class GlobalContextGate(nn.Module):
    """Global gate từ C5: GAP -> FC(ReLU) -> FC -> sigmoid."""

    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)

    def forward(self, c5: torch.Tensor) -> torch.Tensor:
        x = self.pool(c5).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = torch.sigmoid(self.fc2(x))
        return x.unsqueeze(-1).unsqueeze(-1)


class CNNFPNBackbone(nn.Module):
    def __init__(self, base_channels: List[int], fpn_channels: int, fc_dim: int, stem_channels: Optional[List[int]] = None) -> None:
        super().__init__()
        self.backbone = SimpleCNNBackbone(base_channels, stem_channels=stem_channels)
        self.fpn = SimpleFPN(self.backbone.out_channels, fpn_channels)
        self.context = GlobalContextGate(self.backbone.out_channels[-1], fc_dim, fpn_channels)
        self.gate_norms = nn.ModuleList([nn.BatchNorm2d(fpn_channels) for _ in range(6)])
        self.out_channels = [fpn_channels] * 6

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c2, c3, c4, c5 = self.backbone(x)
        pyramid = self.fpn([c2, c3, c4, c5])
        gate = self.context(c5)
        gated = [p * gate for p in pyramid]
        return [bn(feat) for bn, feat in zip(self.gate_norms, gated)]


class AnchorFreeHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 3,
        depthwise: bool = True,
    ) -> None:
        super().__init__()
        def _block():
            if depthwise:
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )
            return ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1)

        cls_tower = []
        reg_tower = []
        for _ in range(num_convs):
            cls_tower.append(_block())
            reg_tower.append(_block())
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, features: List[torch.Tensor]) -> List[dict]:
        outputs = []
        for feat in features:
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)
            outputs.append(
                {
                    "cls_logits": self.cls_logits(cls_feat),
                    "bbox_regression": F.relu(self.reg_pred(reg_feat), inplace=True),
                    "centerness": self.centerness(reg_feat),
                }
            )
        return outputs


class AnchorFreeCNNStudent(nn.Module):
    """Anchor-free student detector: CNN + FPN + FCOS-style head."""

    def __init__(
        self,
        num_classes: int,
        aspect_ratios: Optional[List[List[float]]] = None,
        img_size: int = 512,
        s_min: float = 0.03,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        base_channels: Optional[List[int]] = None,
        fpn_channels: int = 128,
        fc_dim: int = 256,
        stem_channels: Optional[List[int]] = None,
        head_dropout: float = 0.1,
        fcos_strides: Optional[List[int]] = None,
        fcos_ranges: Optional[List[List[float]]] = None,
        head_num_convs: int = 3,
        head_depthwise: bool = True,
    ) -> None:
        super().__init__()
        if base_channels is None:
            base_channels = [32, 64, 128]

        self.backbone = CNNFPNBackbone(base_channels, fpn_channels, fc_dim, stem_channels=stem_channels)
        self.feature_channels = list(self.backbone.out_channels)
        self.num_classes_raw = num_classes
        self.has_background = self.num_classes_raw > 1
        self.num_classes = self.num_classes_raw - 1 if self.has_background else self.num_classes_raw

        self.head = AnchorFreeHead(
            fpn_channels,
            self.num_classes,
            num_convs=head_num_convs,
            depthwise=head_depthwise,
        )
        self.head_drop = nn.Dropout2d(p=head_dropout) if head_dropout > 0 else nn.Identity()

        self.img_size = img_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.fcos_strides = fcos_strides or [4, 8, 16, 32, 64, 128]
        self.fcos_ranges = fcos_ranges or [
            [0, 32],
            [32, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, 1e8],
        ]
        self.anchor_free = True

    def forward_logits(self, images: List[torch.Tensor] | torch.Tensor):
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images, dim=0)
        feats = self.backbone(images)
        feats = [self.head_drop(f) for f in feats]
        head_outs = self.head(feats)

        cls_logits = []
        box_reg = []
        centerness = []
        for out in head_outs:
            n = out["cls_logits"].size(0)
            cls = out["cls_logits"].permute(0, 2, 3, 1).contiguous().view(n, -1, self.num_classes)
            reg = out["bbox_regression"].permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            ctr = out["centerness"].permute(0, 2, 3, 1).contiguous().view(n, -1, 1)
            cls_logits.append(cls)
            box_reg.append(reg)
            centerness.append(ctr)

        return torch.cat(cls_logits, dim=1), torch.cat(box_reg, dim=1), torch.cat(centerness, dim=1), feats

    def forward(self, images: List[torch.Tensor] | torch.Tensor, targets=None):
        cls_logits, box_reg, centerness, feats = self.forward_logits(images)
        if targets is None:
            return self.post_process(cls_logits, box_reg, centerness, feats)
        return self.compute_loss(cls_logits, box_reg, centerness, feats, targets)

    def _generate_points(self, feats: List[torch.Tensor], device: torch.device):
        points_all = []
        for feat, stride in zip(feats, self.fcos_strides):
            h, w = feat.shape[-2:]
            shifts_x = (torch.arange(0, w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(0, h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            points = torch.stack((shift_x, shift_y), dim=-1).view(-1, 2)
            points_all.append(points)
        return points_all

    def _assign_targets(self, points_all, targets, device):
        cls_targets = []
        reg_targets = []
        ctr_targets = []
        pos_mask = []

        for b, tgt in enumerate(targets):
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)
            if self.has_background:
                gt_labels = gt_labels - 1

            cls_b = []
            reg_b = []
            ctr_b = []
            pos_b = []

            for level, points in enumerate(points_all):
                num_points = points.size(0)
                if gt_boxes.numel() == 0:
                    cls_b.append(torch.full((num_points,), -1, device=device, dtype=torch.long))
                    reg_b.append(points.new_zeros((num_points, 4)))
                    ctr_b.append(points.new_zeros((num_points,)))
                    pos_b.append(points.new_zeros((num_points,), dtype=torch.bool))
                    continue

                x = points[:, 0].unsqueeze(1)
                y = points[:, 1].unsqueeze(1)
                l = x - gt_boxes[:, 0].unsqueeze(0)
                t = y - gt_boxes[:, 1].unsqueeze(0)
                r = gt_boxes[:, 2].unsqueeze(0) - x
                b_ = gt_boxes[:, 3].unsqueeze(0) - y
                reg = torch.stack([l, t, r, b_], dim=2)  # (N, M, 4)

                inside_box = (reg.min(dim=2).values > 0)
                max_reg = reg.max(dim=2).values
                range_min, range_max = self.fcos_ranges[level]
                in_range = (max_reg >= range_min) & (max_reg <= range_max)
                valid = inside_box & in_range

                areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                areas = areas.unsqueeze(0).repeat(num_points, 1)
                areas[~valid] = 1e8
                min_area, min_idx = areas.min(dim=1)

                cls = gt_labels[min_idx]
                cls[min_area == 1e8] = -1

                reg_target = reg[torch.arange(num_points, device=device), min_idx]
                pos = cls >= 0

                if pos.any():
                    l_t = reg_target[:, 0]
                    t_t = reg_target[:, 1]
                    r_t = reg_target[:, 2]
                    b_t = reg_target[:, 3]
                    ctr = torch.sqrt(
                        (torch.minimum(l_t, r_t) / torch.maximum(l_t, r_t + 1e-6))
                        * (torch.minimum(t_t, b_t) / torch.maximum(t_t, b_t + 1e-6))
                    )
                else:
                    ctr = reg_target.new_zeros((num_points,))

                cls_b.append(cls)
                reg_b.append(reg_target)
                ctr_b.append(ctr)
                pos_b.append(pos)

            cls_targets.append(torch.cat(cls_b))
            reg_targets.append(torch.cat(reg_b))
            ctr_targets.append(torch.cat(ctr_b))
            pos_mask.append(torch.cat(pos_b))

        return (
            torch.stack(cls_targets),
            torch.stack(reg_targets),
            torch.stack(ctr_targets),
            torch.stack(pos_mask),
        )

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum()

    def compute_loss(self, cls_logits, box_reg, centerness, feats, targets):
        device = cls_logits.device
        points_all = self._generate_points(feats, device)
        cls_t, reg_t, ctr_t, pos = self._assign_targets(points_all, targets, device)

        # Classification focal loss
        B, N, C = cls_logits.shape
        cls_targets = torch.zeros((B, N, C), device=device)
        pos_idx = pos & (cls_t >= 0)
        if pos_idx.any():
            cls_indices = cls_t[pos_idx].long()
            cls_targets[pos_idx, cls_indices] = 1.0
        cls_loss = self._focal_loss(cls_logits, cls_targets) / max(1, pos.sum().item())

        # Regression GIoU loss
        if pos_idx.any():
            points = torch.cat(points_all).to(device)
            points = points.unsqueeze(0).expand(B, -1, 2)
            pts = points[pos_idx]
            pred_reg = box_reg[pos_idx]
            gt_reg = reg_t[pos_idx]

            pred_boxes = torch.stack(
                [
                    pts[:, 0] - pred_reg[:, 0],
                    pts[:, 1] - pred_reg[:, 1],
                    pts[:, 0] + pred_reg[:, 2],
                    pts[:, 1] + pred_reg[:, 3],
                ],
                dim=1,
            )
            gt_boxes = torch.stack(
                [
                    pts[:, 0] - gt_reg[:, 0],
                    pts[:, 1] - gt_reg[:, 1],
                    pts[:, 0] + gt_reg[:, 2],
                    pts[:, 1] + gt_reg[:, 3],
                ],
                dim=1,
            )
            giou = generalized_iou(pred_boxes, gt_boxes)
            reg_loss = (1.0 - giou).sum() / max(1, pos.sum().item())
        else:
            reg_loss = cls_logits.new_zeros(())

        # Centerness loss
        if pos_idx.any():
            ctr_pred = centerness[pos_idx].squeeze(1)
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ctr_t[pos_idx], reduction="sum")
            ctr_loss = ctr_loss / max(1, pos.sum().item())
        else:
            ctr_loss = cls_logits.new_zeros(())

        return {"classification": cls_loss, "bbox_regression": reg_loss, "centerness": ctr_loss}

    def post_process(
        self,
        cls_logits: torch.Tensor,
        box_reg: torch.Tensor,
        centerness: torch.Tensor,
        feats: Optional[List[torch.Tensor]] = None,
        points_all: Optional[List[torch.Tensor]] = None,
        score_thresh: Optional[float] = None,
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
        if points_all is None:
            if feats is None:
                raise ValueError("Either feats or points_all must be provided for anchor-free post_process.")
            points_all = self._generate_points(feats, device)
        points = torch.cat(points_all).to(device)

        scores = torch.sigmoid(cls_logits) * torch.sigmoid(centerness)

        outputs = []
        for b in range(cls_logits.size(0)):
            score_b, labels = scores[b].max(dim=1)
            keep = score_b > score_thresh
            if not keep.any():
                outputs.append(torch.zeros((0, 6), device=device))
                continue

            score_b = score_b[keep]
            labels = labels[keep]
            reg = box_reg[b][keep]
            pts = points[keep]

            boxes = torch.stack(
                [
                    pts[:, 0] - reg[:, 0],
                    pts[:, 1] - reg[:, 1],
                    pts[:, 0] + reg[:, 2],
                    pts[:, 1] + reg[:, 3],
                ],
                dim=1,
            )

            if pre_nms_topk > 0 and boxes.size(0) > pre_nms_topk:
                topk_idx = torch.topk(score_b, k=pre_nms_topk).indices
                boxes = boxes[topk_idx]
                score_b = score_b[topk_idx]
                labels = labels[topk_idx]

            if self.has_background:
                labels = labels + 1

            keep_idx = batched_nms(boxes, score_b, labels, self.nms_thresh)
            boxes = boxes[keep_idx]
            score_b = score_b[keep_idx]
            labels = labels[keep_idx]

            preds = torch.cat([boxes, score_b.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
            if max_detections > 0 and preds.size(0) > max_detections:
                top_idx = torch.topk(preds[:, 4], k=max_detections).indices
                preds = preds[top_idx]
            outputs.append(preds)

        return outputs
