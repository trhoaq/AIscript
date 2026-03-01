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

def calculate_stats(preds: List[torch.Tensor], targets: List[dict], iou_threshold: float = 0.5):
    """
    Highly optimized calculation of True Positives and scores for mAP, using GPU if available.
    """
    import numpy as np
    from torchvision.ops import box_iou
    
    if not preds: return {}
    device = preds[0].device

    # 1. Pre-group Ground Truths by class and image index
    gt_by_cls_img = {} # {label: {img_idx: boxes_tensor}}
    unique_labels = set()
    
    for i, t in enumerate(targets):
        labels = t['labels']
        boxes = t['boxes'].to(device) # Keep GTs on the same device as preds
        for label_tensor in labels.unique():
            label = label_tensor.item()
            unique_labels.add(label)
            if label not in gt_by_cls_img: gt_by_cls_img[label] = {}
            mask = (labels == label)
            gt_by_cls_img[label][i] = boxes[mask]

    stats = {}
    for label in unique_labels:
        # 2. Collect all predictions for this class
        cls_preds_list = []
        total_gt = 0
        
        if label in gt_by_cls_img:
            for img_idx in gt_by_cls_img[label]:
                total_gt += len(gt_by_cls_img[label][img_idx])

        for i, p in enumerate(preds):
            if p.shape[0] == 0: continue
            mask = (p[:, 5] == label)
            cls_p = p[mask]
            if cls_p.shape[0] > 0:
                img_indices = torch.full((cls_p.shape[0], 1), i, device=device)
                combined = torch.cat([cls_p[:, 4:5], img_indices, cls_p[:, :4]], dim=1)
                cls_preds_list.append(combined)

        if total_gt == 0 or not cls_preds_list:
            continue

        # 3. Sort global predictions by confidence
        cls_preds = torch.cat(cls_preds_list, dim=0)
        sort_idx = torch.argsort(cls_preds[:, 0], descending=True)
        cls_preds = cls_preds[sort_idx]

        # 4. Match with GTs (Core matching loop)
        tp = []
        conf = []
        
        matched_mask = {img_idx: torch.zeros(len(boxes), dtype=torch.bool, device=device) 
                        for img_idx, boxes in gt_by_cls_img[label].items()}

        for pred in cls_preds:
            score, img_idx, p_box = pred[0].item(), int(pred[1].item()), pred[2:].unsqueeze(0)
            conf.append(score)
            
            is_tp = 0
            if img_idx in gt_by_cls_img[label]:
                gt_boxes = gt_by_cls_img[label][img_idx]
                # box_iou on GPU is extremely fast
                ious = box_iou(p_box, gt_boxes)
                best_iou, best_gt_idx = ious.max(1)
                
                if best_iou.item() > iou_threshold:
                    if not matched_mask[img_idx][best_gt_idx]:
                        is_tp = 1
                        matched_mask[img_idx][best_gt_idx] = True
            
            tp.append(is_tp)
            
        stats[label] = {'tp': tp, 'conf': conf, 'total_gt': total_gt}

    return stats

def compute_metrics(stats):
    """Computes mAP@0.5, overall Precision and Recall from gathered stats."""
    import numpy as np
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for label, data in stats.items():
        tp = np.array(data['tp'])
        total_gt += data['total_gt']
        
        if len(tp) == 0:
            aps.append(0)
            continue
            
        fp = 1 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        recall = tp_cum / (data['total_gt'] + 1e-16)
        
        total_tp += tp.sum()
        total_fp += fp.sum()

        # VOC AP calculation (11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
            else:
                p = 0
            ap += p / 11
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0
    overall_precision = total_tp / (total_tp + total_fp + 1e-16)
    overall_recall = total_tp / (total_gt + 1e-16)
    
    return mAP, overall_precision, overall_recall


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
