from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ssdlite_mobilenet import SSDMobile, SSDLiteHead


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
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        out = out + x
        return F.relu(out, inplace=True)


class SpatialPyramidPooling(nn.Module):
    """SPP layer with stride-2 maxpools to replace downsampling maxpool."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=7, stride=1, padding=4),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(self.pools) + 1), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]+[p(x) for p in self.pools]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class SimpleCNNBackbone(nn.Module):
    """
    Simple CNN backbone with preprocessing stages:
    S1 (stride 2) -> S2 (stride 2) -> C3 (stride 8) -> C4 (stride 16) -> C5 (stride 32)
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

        self.out_channels = [c1, c2, c3]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stage_s1(x)
        x = self.stage_s2(x)

        c3 = self.res_c3(self.conv_c3(x))
        c4 = self.res_c4(self.conv_c4(c3))
        c5 = self.res_c5(self.conv_c5(c4))
        c5 = self.spp_c5(c5)
        return [c3, c4, c5]


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
        c3, c4, c5 = features

        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)

        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(F.relu(p6, inplace=True))

        return [p3, p4, p5, p6, p7]


class GlobalContextGate(nn.Module):
    """Per-level gate: mỗi FPN level có gate riêng từ pooling của chính nó."""

    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_levels: int) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
            # Mỗi level output 1 vector gate riêng                                                          self.fc2 = nn.Linear(hidden_dim, out_channels * num_levels)
        self.out_channels = out_channels

    def forward(self, c5: torch.Tensor) -> List[torch.Tensor]:                                                                   
        x = self.pool(c5).flatten(1)          # (B, C)
        x = F.relu(self.fc1(x), inplace=True) # (B, hidden)
        x = torch.sigmoid(self.fc2(x))        # (B, out_channels * num_lev                                            # Tách thành list gate per level
        gates = x.chunk(self.num_levels, dim=1)
        return [g.unsqueeze(-1).unsqueeze(-1) for g in gates]  # [(B,C,1,1), ...]


class CNNFPNBackbone(nn.Module):
    def __init__(self, base_channels: List[int], fpn_channels: int, fc_dim: int, stem_channels: Optional[List[int]] = None) -> None:
        super().__init__()
        self.backbone = SimpleCNNBackbone(base_channels, stem_channels=stem_channels)
        self.fpn = SimpleFPN(self.backbone.out_channels, fpn_channels)
        self.context = GlobalContextGate(self.backbone.out_channels[-1], fc_dim, fpn_channels, num_levels=5)
        self.gate_norms = nn.ModuleList([nn.BatchNorm2d(fpn_channels) for _ in range(5)])
        self.out_channels = [fpn_channels] * 5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        pyramid = self.fpn([c3, c4, c5])
        gate = self.context(c5)
        gated = [p * g for p, g in zip(pyramid, gate)]
        return [bn(feat) for bn, feat in zip(self.gate_norms, gated)]


class SSDLiteHeadWithDropout(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.head = SSDLiteHead(in_channels, num_anchors, num_classes)

    def forward(self, features: List[torch.Tensor]) -> dict:
        dropped = [self.drop(f) for f in features]
        return self.head(dropped)


class SSDCNNStudent(SSDMobile):
    """
    Student detection model: Simple CNN + FPN + SSDLite head.
    Uses batch NMS via inherited post_process.
    No pretrained backbone loading.
    """

    def __init__(
        self,
        num_classes: int,
        aspect_ratios: Optional[List[List[float]]] = None,
        img_size: int = 512,
        s_min: float = 0.07,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        base_channels: Optional[List[int]] = None,
        fpn_channels: int = 128,
        fc_dim: int = 256,
        stem_channels: Optional[List[int]] = None,
        head_dropout: float = 0.1,
    ) -> None:
        if aspect_ratios is None:
            aspect_ratios = [[2], [2, 3], [2, 3], [2], [2]]
        if base_channels is None:
            base_channels = [32, 64, 128]

        super().__init__(
            num_classes=num_classes,
            aspect_ratios=aspect_ratios,
            img_size=img_size,
            s_min=s_min,
            s_max=s_max,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            pretrained_backbone=False,
            pretrained_backbone_model_name="",
            width_mult=1.0,
        )

        self.backbone = CNNFPNBackbone(base_channels, fpn_channels, fc_dim, stem_channels=stem_channels)
        self.feature_channels = list(self.backbone.out_channels)
        self.head = SSDLiteHeadWithDropout(
            self.feature_channels,
            self.num_anchors,
            num_classes,
            dropout=head_dropout,
        )
        self.pretrained_backbone = False
        self.backbone_has_weights_loaded = True

    def load_pretrained_weights(self, device: torch.device):
        print("Pretrained backbone disabled for SSDCNNStudent. Skipping weight loading.")
