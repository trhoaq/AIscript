from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class HardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class HardSwish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hsig = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsig(x)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        groups: int = 1,
        act: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeeze_ch = _make_divisible(in_ch / squeeze_factor, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, kernel_size=1)
        self.gate = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        expansion: float,
        act: nn.Module,
        use_se: bool,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch

        exp_ch = _make_divisible(in_ch * expansion, 8)
        layers: List[nn.Module] = []

        if exp_ch != in_ch:
            layers.append(ConvBNAct(in_ch, exp_ch, kernel=1, stride=1, act=act))

        layers.append(ConvBNAct(exp_ch, exp_ch, kernel=kernel, stride=stride, groups=exp_ch, act=act))

        if use_se:
            layers.append(SqueezeExcite(exp_ch, squeeze_factor=4))

        layers.append(ConvBNAct(exp_ch, out_ch, kernel=1, stride=1, act=nn.Identity()))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cfgs: List[tuple],
        width_mult: float = 1.0,
        final_ch: int = 1280,
        first_ch: int = 16,
        last_conv_ch: Optional[int] = None,
        classifier_act: Optional[nn.Module] = None,
        finegrain_classification_mode: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        first_ch = _make_divisible(first_ch * width_mult, 8)
        self.stem = ConvBNAct(3, first_ch, kernel=3, stride=2, act=HardSwish())

        layers: List[nn.Module] = []
        in_ch = first_ch
        for (k, exp, out, se, act_name, s) in cfgs:
            out_ch = _make_divisible(out * width_mult, 8)
            act = nn.ReLU(inplace=True) if act_name == "relu" else HardSwish()
            layers.append(InvertedResidual(in_ch, out_ch, k, s, exp, act, se))
            in_ch = out_ch

        if last_conv_ch is None:
            last_conv_ch = 960 if final_ch == 1280 else final_ch
        last_conv_ch = _make_divisible(last_conv_ch * width_mult, 8)

        layers.append(ConvBNAct(in_ch, last_conv_ch, kernel=1, stride=1, act=HardSwish()))
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        if finegrain_classification_mode:
            final_out = final_ch
        else:
            final_out = _make_divisible(final_ch * width_mult, 8)
        classifier_act = classifier_act if classifier_act is not None else HardSwish()
        self.classifier = nn.Sequential(
            nn.Conv2d(last_conv_ch, final_out, kernel_size=1),
            classifier_act,
            nn.Flatten(1),
            nn.Linear(final_out, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def mobilenet_v3_large(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    finegrain_classification_mode: bool = False,
) -> MobileNetV3:
    cfgs = [
        (3, 1, 16, False, "relu", 1),
        (3, 4, 24, False, "relu", 2),
        (3, 3, 24, False, "relu", 1),
        (5, 3, 40, True, "relu", 2),
        (5, 3, 40, True, "relu", 1),
        (5, 3, 40, True, "relu", 1),
        (3, 6, 80, False, "hswish", 2),
        (3, 2.5, 80, False, "hswish", 1),
        (3, 184 / 80.0, 80, False, "hswish", 1),
        (3, 184 / 80.0, 80, False, "hswish", 1),
        (3, 6, 112, True, "hswish", 1),
        (3, 6, 112, True, "hswish", 1),
        (5, 6, 160, True, "hswish", 2),
        (5, 6, 160, True, "hswish", 1),
        (5, 6, 160, True, "hswish", 1),
    ]
    return MobileNetV3(
        num_classes=num_classes,
        cfgs=cfgs,
        width_mult=width_mult,
        final_ch=1280,
        first_ch=16,
        last_conv_ch=960,
        classifier_act=HardSwish(),
        finegrain_classification_mode=finegrain_classification_mode,
    )


def mobilenet_v3_small(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    finegrain_classification_mode: bool = False,
) -> MobileNetV3:
    cfgs = [
        (3, 1, 16, True, "relu", 2),
        (3, 72 / 16.0, 24, False, "relu", 2),
        (3, 88 / 24.0, 24, False, "relu", 1),
        (5, 4, 40, True, "hswish", 2),
        (5, 6, 40, True, "hswish", 1),
        (5, 6, 40, True, "hswish", 1),
        (5, 3, 48, True, "hswish", 1),
        (5, 3, 48, True, "hswish", 1),
        (5, 6, 96, True, "hswish", 2),
        (5, 6, 96, True, "hswish", 1),
        (5, 6, 96, True, "hswish", 1),
    ]
    return MobileNetV3(
        num_classes=num_classes,
        cfgs=cfgs,
        width_mult=width_mult,
        final_ch=1024,
        first_ch=16,
        last_conv_ch=576,
        classifier_act=HardSwish(),
        finegrain_classification_mode=finegrain_classification_mode,
    )


def mobilenet_v3_large_minimalistic(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    finegrain_classification_mode: bool = False,
) -> MobileNetV3:
    cfgs = [
        (3, 1, 16, False, "relu", 1),
        (3, 4, 24, False, "relu", 2),
        (3, 3, 24, False, "relu", 1),
        (3, 3, 40, False, "relu", 2),
        (3, 3, 40, False, "relu", 1),
        (3, 3, 40, False, "relu", 1),
        (3, 6, 80, False, "relu", 2),
        (3, 2.5, 80, False, "relu", 1),
        (3, 184 / 80.0, 80, False, "relu", 1),
        (3, 184 / 80.0, 80, False, "relu", 1),
        (3, 6, 112, False, "relu", 1),
        (3, 6, 112, False, "relu", 1),
        (3, 6, 160, False, "relu", 2),
        (3, 6, 160, False, "relu", 1),
        (3, 6, 160, False, "relu", 1),
    ]
    return MobileNetV3(
        num_classes=num_classes,
        cfgs=cfgs,
        width_mult=width_mult,
        final_ch=1280,
        first_ch=16,
        last_conv_ch=960,
        classifier_act=nn.ReLU(inplace=True),
        finegrain_classification_mode=finegrain_classification_mode,
    )


def mobilenet_v3_small_minimalistic(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    finegrain_classification_mode: bool = False,
) -> MobileNetV3:
    cfgs = [
        (3, 1, 16, False, "relu", 2),
        (3, 72 / 16.0, 24, False, "relu", 2),
        (3, 88 / 24.0, 24, False, "relu", 1),
        (3, 4, 40, False, "relu", 2),
        (3, 6, 40, False, "relu", 1),
        (3, 6, 40, False, "relu", 1),
        (3, 3, 48, False, "relu", 1),
        (3, 3, 48, False, "relu", 1),
        (3, 6, 96, False, "relu", 2),
        (3, 6, 96, False, "relu", 1),
        (3, 6, 96, False, "relu", 1),
    ]
    return MobileNetV3(
        num_classes=num_classes,
        cfgs=cfgs,
        width_mult=width_mult,
        final_ch=1024,
        first_ch=16,
        last_conv_ch=576,
        classifier_act=nn.ReLU(inplace=True),
        finegrain_classification_mode=finegrain_classification_mode,
    )


__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mobilenet_v3_large_minimalistic",
    "mobilenet_v3_small_minimalistic",
    "load_pretrained_from_timm",
]


_WEIGHT_MODULES: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)


def _collect_weight_modules(model: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    for m in model.modules():
        if isinstance(m, _WEIGHT_MODULES):
            modules.append(m)
    return modules


def _copy_module_weights(src: nn.Module, dst: nn.Module) -> bool:
    if isinstance(src, nn.Conv2d) and isinstance(dst, nn.Conv2d):
        if src.weight.shape != dst.weight.shape:
            return False
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)
        return True
    if isinstance(src, nn.BatchNorm2d) and isinstance(dst, nn.BatchNorm2d):
        if src.weight.shape != dst.weight.shape:
            return False
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)
        dst.running_mean.data.copy_(src.running_mean.data)
        dst.running_var.data.copy_(src.running_var.data)
        if hasattr(src, "num_batches_tracked") and hasattr(dst, "num_batches_tracked"):
            dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)
        return True
    if isinstance(src, nn.Linear) and isinstance(dst, nn.Linear):
        if src.weight.shape != dst.weight.shape:
            return False
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)
        return True
    return False


def load_pretrained_from_timm(
    model: nn.Module,
    timm_name: str,
    verbose: bool = True,
) -> nn.Module:
    """
    Load pretrained weights from a timm MobileNetV3 model into this implementation.
    The transfer is done by module order + shape matching. Classifier weights are
    skipped automatically if num_classes differs.
    """
    import timm

    src_model = timm.create_model(timm_name, pretrained=True)
    src_modules = _collect_weight_modules(src_model)
    dst_modules = _collect_weight_modules(model)

    copied = 0
    skipped = 0
    total = min(len(src_modules), len(dst_modules))
    for i in range(total):
        if _copy_module_weights(src_modules[i], dst_modules[i]):
            copied += 1
        else:
            skipped += 1

    if verbose:
        msg = (
            f"timm -> custom: copied {copied} modules, skipped {skipped}. "
            f"src={len(src_modules)} dst={len(dst_modules)}"
        )
        print(msg)

    return model
