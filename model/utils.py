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
    Calculate True Positives, False Positives and total Ground Truths.
    preds: List of tensors [N, 6] (x1, y1, x2, y2, score, label)
    targets: List of dicts {'boxes': [M, 4], 'labels': [M]}
    """
    import numpy as np
    from torchvision.ops import box_iou
    
    # Collect all unique labels from targets
    all_target_labels = torch.cat([t['labels'] for t in targets])
    unique_labels = all_target_labels.unique()
    
    stats = {}

    for label in unique_labels:
        label = label.item()
        cls_gts = 0
        global_cls_preds = []
        
        # 1. Collect all preds and total gts for this class
        for i in range(len(preds)):
            p = preds[i]
            t = targets[i]
            
            # GTs count
            cls_gts += (t['labels'] == label).sum().item()
            
            # Filter preds for this class
            mask_p = p[:, 5] == label
            for box_info in p[mask_p]:
                global_cls_preds.append({
                    'box': box_info[:4].unsqueeze(0),
                    'conf': box_info[4].item(),
                    'img_idx': i
                })
        
        if cls_gts == 0: continue
        stats[label] = {'tp': [], 'conf': [], 'total_gt': cls_gts}
        
        if not global_cls_preds:
            continue

        # 2. Sort global predictions by confidence
        global_cls_preds.sort(key=lambda x: x['conf'], reverse=True)
        
        # 3. Match with GTs
        # Track matched GTs per image to avoid double counting
        image_gts_matched = {} # {img_idx: bitmask_tensor}
        for i in range(len(targets)):
            num_gts_in_img = (targets[i]['labels'] == label).sum().item()
            image_gts_matched[i] = torch.zeros(num_gts_in_img, device=preds[0].device)

        for p_item in global_cls_preds:
            img_idx = p_item['img_idx']
            p_box = p_item['box']
            
            gt_boxes_img = targets[img_idx]['boxes'][targets[img_idx]['labels'] == label]
            
            is_tp = 0
            if len(gt_boxes_img) > 0:
                ious = box_iou(p_box, gt_boxes_img)
                best_iou, best_gt_idx = ious.max(1)
                
                if best_iou.item() > iou_threshold:
                    if image_gts_matched[img_idx][best_gt_idx] == 0:
                        is_tp = 1
                        image_gts_matched[img_idx][best_gt_idx] = 1
            
            stats[label]['tp'].append(is_tp)
            stats[label]['conf'].append(p_item['conf'])

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
