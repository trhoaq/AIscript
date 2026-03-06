from model.ssdlite_mobilenet import SSDMobile


class SSDGhost(SSDMobile):
    """Backward-compatible name: student now uses SSDLite MobileNetV3 architecture."""

    def __init__(
        self,
        num_classes,
        width=1.0,
        img_size=320,
        score_thresh=0.05,
        nms_thresh=0.45,
        pretrained_backbone=False,
        pretrained_backbone_model_name="mobilenetv3_large_100",
    ):
        super().__init__(
            num_classes=num_classes,
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            img_size=img_size,
            s_min=0.07,
            s_max=0.95,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            pretrained_backbone=pretrained_backbone,
            pretrained_backbone_model_name=pretrained_backbone_model_name,
            width_mult=width,
        )
