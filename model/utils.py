from typing import List, Tuple
import torch

# --- Bounding Box Utilities ---

def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (xmin, ymin, xmax, ymax) to (center_x, center_y, width, height)."""
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)."""
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)

# --- Anchor Generation ---

class DefaultBoxGenerator:
    """
    Generates default anchor boxes for SSD.
    This is a generic implementation that can be shared across different SSD models.
    """
    def __init__(
        self,
        aspect_ratios: List[List[float]],
        s_min: float = 0.1,
        s_max: float = 0.9,
    ) -> None:
        self.aspect_ratios = aspect_ratios
        self.s_min = s_min
        self.s_max = s_max

    def num_anchors_per_location(self) -> List[int]:
        """Returns the number of anchors per feature map level."""
        return [2 + 2 * len(ars) for ars in self.aspect_ratios]

    def _scales(self, m: int) -> List[float]:
        """Generates scales for the anchor boxes."""
        if m == 1: 
            return [self.s_max]
        return [self.s_min + (self.s_max - self.s_min) * k / (m - 1) for k in range(m)]

    def generate(self, feat_sizes: List[Tuple[int, int]], img_size: int, device: torch.device) -> torch.Tensor:
        """
        Generates anchor boxes for all feature maps.
        Args:
            feat_sizes: A list of (height, width) for each feature map.
            img_size: The size of the input image.
            device: The torch device to create tensors on.
        Returns:
            A tensor of shape (num_anchors, 4) with boxes in (xmin, ymin, xmax, ymax) format.
        """
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
                    # Aspect ratio 1, size sk
                    priors.append(torch.tensor([cx, cy, sk, sk], device=device))
                    # Aspect ratio 1, size sqrt(sk*sk1)
                    s_prime = (sk * sk1) ** 0.5
                    priors.append(torch.tensor([cx, cy, s_prime, s_prime], device=device))
                    # Other aspect ratios
                    for ar in self.aspect_ratios[k]:
                        r = ar ** 0.5
                        priors.append(torch.tensor([cx, cy, sk * r, sk / r], device=device))
                        priors.append(torch.tensor([cx, cy, sk / r, sk * r], device=device))
        
        priors_t = torch.stack(priors, dim=0)
        priors_t.clamp_(0.0, 1.0)
        
        # Convert to (xmin, ymin, xmax, ymax) and scale to image size
        return _cxcywh_to_xyxy(priors_t) * img_size
