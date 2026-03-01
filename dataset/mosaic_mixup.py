import torch
import random
import numpy as np
from torch.utils.data import Dataset

class MosaicMixupDataset(Dataset):
    """
    A wrapper dataset to apply Mosaic and MixUp augmentations.
    """
    def __init__(self, base_dataset, img_size, p_mosaic=0.5, p_mixup=0.5, transform=None):
        self.base_dataset = base_dataset
        self.img_size = img_size
        self.p_mosaic = p_mosaic
        self.p_mixup = p_mixup
        self.final_transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        use_mosaic = random.random() < self.p_mosaic
        use_mixup = random.random() < self.p_mixup

        if use_mosaic:
            img, target = self._load_mosaic(index)
        elif use_mixup:
            img, target = self._load_mixup(index)
        else:
            img, target = self.base_dataset[index]

        if img is None or target is None: 
            return None, None

        if self.final_transform:
            transformed = self.final_transform(image=img, bboxes=target["boxes"], class_labels=target["labels"])
            img = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        return img, target

    def _load_mosaic(self, index):
        indices = [index] + [random.randint(0, len(self.base_dataset) - 1) for _ in range(3)]
        
        mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 128, dtype=np.uint8)
        center_x, center_y = [int(random.uniform(self.img_size * 0.5, self.img_size * 1.5)) for _ in range(2)]
        
        mosaic_boxes, mosaic_labels = [], []

        for i, idx in enumerate(indices):
            img_i, target_i = self.base_dataset[idx]
            if img_i is None: continue # Skip if a sample is invalid
            
            h, w, _ = img_i.shape
            
            if i == 0: # top-left
                x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1: # top-right
                x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, self.img_size * 2), center_y
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: # bottom-left
                x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(self.img_size * 2, center_y + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else: # bottom-right
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, self.img_size * 2), min(center_y + h, self.img_size * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            for box, label in zip(target_i["boxes"], target_i["labels"]):
                x_min, y_min, x_max, y_max = box
                new_box = [x_min + padw, y_min + padh, x_max + padw, y_max + padh]
                mosaic_boxes.append(new_box)
                mosaic_labels.append(label)

        # Cut out the final mosaic image
        final_img = mosaic_img[center_y - self.img_size // 2 : center_y + self.img_size // 2, 
                               center_x - self.img_size // 2 : center_x + self.img_size // 2]
        
        # Clip boxes to the final image size
        final_boxes, final_labels = [], []
        for box, label in zip(mosaic_boxes, mosaic_labels):
            x_min, y_min, x_max, y_max = box
            x_min_adj = x_min - (center_x - self.img_size // 2)
            y_min_adj = y_min - (center_y - self.img_size // 2)
            x_max_adj = x_max - (center_x - self.img_size // 2)
            y_max_adj = y_max - (center_y - self.img_size // 2)

            clipped_box = [
                max(0, x_min_adj), max(0, y_min_adj),
                min(self.img_size, x_max_adj), min(self.img_size, y_max_adj)
            ]
            
            # Filter out boxes that are too small
            if clipped_box[2] > clipped_box[0] and clipped_box[3] > clipped_box[1]:
                final_boxes.append(clipped_box)
                final_labels.append(label)

        return final_img, {"boxes": final_boxes, "labels": final_labels}

    def _load_mixup(self, index):
        img1, target1 = self.base_dataset[index]
        idx2 = random.randint(0, len(self.base_dataset) - 1)
        img2, target2 = self.base_dataset[idx2]

        if img1 is None or img2 is None: return self.base_dataset[index]

        r = np.random.beta(1.5, 1.5)
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        boxes = target1["boxes"] + target2["boxes"]
        labels = target1["labels"] + target2["labels"]
        
        return img, {"boxes": boxes, "labels": labels}
