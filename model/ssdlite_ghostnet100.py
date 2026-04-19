import gc
from typing import List, Optional

import torch
import torch.nn as nn

from .ghostnet import ghostnetv3
from .mobilenetv3_torch import load_pretrained_from_timm
from .ssdlite_mobilenet import ConvBNReLU6, DepthwiseSeparableConv, SSDMobile, SSDLiteHead


def _make_divisible(v: float, divisor: int = 4, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class GhostNetV3SSDLiteBackbone(nn.Module):
    """GhostNetV3 backbone adapted for SSDLite feature extraction."""

    def __init__(self, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = float(width_mult)

        # Build local GhostNetV3 classifier and reuse stem/blocks for detector features.
        self.backbone_model = ghostnetv3(width=self.width_mult)

        # Feature map selection points for 320x320 input:
        # - idx 6 -> 20x20 (112 channels for 1.0x)
        # - idx 9 -> 10x10 (960 channels for 1.0x)
        self.low_level_index = 6
        self.high_level_index = 9

        c_low = _make_divisible(112 * self.width_mult, 4)
        c_high = _make_divisible(960 * self.width_mult, 4)

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
        x = self.backbone_model.conv_stem(x)
        x = self.backbone_model.bn1(x)
        x = self.backbone_model.act1(x)

        low_level: Optional[torch.Tensor] = None
        high_level: Optional[torch.Tensor] = None

        for idx, block in enumerate(self.backbone_model.blocks):
            x = block(x)
            if idx == self.low_level_index:
                low_level = x
            if idx == self.high_level_index:
                high_level = x

        if low_level is None or high_level is None:
            raise RuntimeError("Failed to extract GhostNetV3 SSD feature maps.")

        p3 = self.proj_low(low_level)
        p4 = self.proj_high(high_level)
        p5 = self.extra_1(p4)
        p6 = self.extra_2(p5)
        p7 = self.extra_3(p6)
        p8 = self.extra_4(p7)

        return [p3, p4, p5, p6, p7, p8]


class SSDGhostNetV3(SSDMobile):
    """
    SSDLite student model with GhostNetV3 1.0x backbone.
    Reuses SSDMobile head/loss/post-process improvements from teacher architecture.
    """

    def __init__(
        self,
        num_classes: int,
        aspect_ratios: Optional[List[List[float]]] = None,
        img_size: int = 320,
        s_min: float = 0.07,
        s_max: float = 0.95,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        pretrained_backbone: bool = True,
        pretrained_backbone_model_name: str = "ghostnetv3_100.in1k",
        width_mult: float = 1.0,
    ) -> None:
        if aspect_ratios is None:
            aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        super().__init__(
            num_classes=num_classes,
            aspect_ratios=aspect_ratios,
            img_size=img_size,
            s_min=s_min,
            s_max=s_max,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            pretrained_backbone=pretrained_backbone,
            pretrained_backbone_model_name=pretrained_backbone_model_name,
            width_mult=1.0,
        )

        self.backbone = GhostNetV3SSDLiteBackbone(width_mult=width_mult)
        self.feature_channels = list(self.backbone.out_channels)
        self.head = SSDLiteHead(self.feature_channels, self.num_anchors, num_classes)
        self.pretrained_backbone_model_name = str(pretrained_backbone_model_name).strip() or "ghostnetv3_100.in1k"
        self.backbone_has_weights_loaded = False

    def load_pretrained_weights(self, device: torch.device):
        if not self.pretrained_backbone:
            print("Pretrained backbone disabled. Skipping weight loading.")
            return

        if self.backbone_has_weights_loaded:
            print("Pretrained weights already loaded for SSDGhostNetV3 backbone.")
            return

        model_name = self.pretrained_backbone_model_name
        print(
            f"Loading GhostNetV3 backbone ImageNet-1k pretrained weights "
            f"from timm model '{model_name}' to device: {device}"
        )
        try:
            with torch.no_grad():
                self.backbone.load_imagenet_pretrained(model_name)
            self.backbone_has_weights_loaded = True
            print("Successfully loaded GhostNetV3 backbone from timm ImageNet-1k weights.")
        except ImportError:
            print("Failed to import timm. Install with: pip install timm")
            self.backbone_has_weights_loaded = False
        except Exception as e:
            print(f"Failed to load GhostNetV3 timm ImageNet-1k weights: {e}")
            self.backbone_has_weights_loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SSDGhostNet100(SSDGhostNetV3):
    """Backward-compatible alias."""
