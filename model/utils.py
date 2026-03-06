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


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in (xmin, ymin, xmax, ymax) format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    union = area1 + area2 - inter
    return inter / (union + 1e-16)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Non-maximum suppression implemented with native PyTorch ops."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-16)

        remain = torch.where(iou <= iou_threshold)[0]
        order = rest[remain]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def calculate_stats(preds: List[torch.Tensor], targets: List[dict], iou_threshold: float = 0.5):
    """
    Super optimized calculation of TP/FP using vectorized operations and efficient CPU matching.
    """
    import numpy as np
    
    if not preds: return {}
    
    # 1. Concatenate all predictions into one big tensor [Total_N, 7]
    # Each row: [x1, y1, x2, y2, score, label, img_idx]
    all_preds_list = []
    for i, p in enumerate(preds):
        if p.shape[0] > 0:
            img_idx = torch.full((p.shape[0], 1), i, device=p.device, dtype=p.dtype)
            all_preds_list.append(torch.cat([p, img_idx], dim=1))
    
    if not all_preds_list: return {}
    all_preds = torch.cat(all_preds_list, dim=0).detach().cpu() # Move to CPU for sequential matching logic
    
    # 2. Pre-index Ground Truths by label and image_idx on CPU
    gt_indexed = {} # {label: {img_idx: boxes}}
    unique_labels = set()
    for i, t in enumerate(targets):
        l_np = t['labels'].cpu().numpy()
        b_np = t['boxes'].cpu().numpy()
        for label in np.unique(l_np):
            unique_labels.add(int(label))
            if label not in gt_indexed: gt_indexed[label] = {}
            gt_indexed[label][i] = b_np[l_np == label]

    stats = {}
    for label in unique_labels:
        # 3. Filter predictions for this class
        cls_mask = all_preds[:, 5] == label
        cls_preds = all_preds[cls_mask]
        if cls_preds.shape[0] == 0: continue
        
        # Sort by confidence
        cls_preds = cls_preds[torch.argsort(cls_preds[:, 4], descending=True)]
        
        # Count total GTs
        total_gt = sum(len(v) for v in gt_indexed.get(label, {}).values())
        if total_gt == 0: continue

        # 4. Matching logic (Sequential but fast on CPU/NumPy)
        tp = np.zeros(len(cls_preds))
        conf = cls_preds[:, 4].numpy()
        
        # Track matched GTs per image
        matched_gts = {img_idx: np.zeros(len(boxes), dtype=bool) 
                       for img_idx, boxes in gt_indexed.get(label, {}).items()}

        preds_np = cls_preds.numpy() # [x1, y1, x2, y2, score, label, img_idx]
        
        for i in range(len(preds_np)):
            img_idx = int(preds_np[i, 6])
            if img_idx not in matched_gts:
                continue
                
            pred_box = preds_np[i, :4]
            gt_boxes = gt_indexed[label][img_idx]
            
            # Fast IoU calculation for one pred box vs all GT boxes in that image
            # Manual IoU to avoid torch overhead
            ixmin = np.maximum(gt_boxes[:, 0], pred_box[0])
            iymin = np.maximum(gt_boxes[:, 1], pred_box[1])
            ixmax = np.minimum(gt_boxes[:, 2], pred_box[2])
            iymax = np.minimum(gt_boxes[:, 3], pred_box[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            
            uni = ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) -
                   inters)
            
            overlaps = inters / uni
            best_idx = np.argmax(overlaps)
            if overlaps[best_idx] > iou_threshold:
                if not matched_gts[img_idx][best_idx]:
                    tp[i] = 1
                    matched_gts[img_idx][best_idx] = True
        
        stats[label] = {'tp': tp.tolist(), 'conf': conf.tolist(), 'total_gt': total_gt}

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
