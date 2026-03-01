import os
import cv2
import xml.etree.ElementTree as ET
import json
import torch
from torch.utils.data import Dataset, ConcatDataset
import albumentations as A
import numpy as np # Move numpy import to the top

# --- Configuration & Mapping ---
try:
    with open("config.json", "r", encoding="utf-8") as f:
        CFG = json.load(f)
    OBJ_CLASSES = CFG["obj_classes"]
except Exception:
    OBJ_CLASSES = ["background", "target"] # Fallback

class_to_idx = {cls: i for i, cls in enumerate(OBJ_CLASSES)}

class PascalVOCDataset(Dataset):
    """
    Base dataset for loading images and annotations from Pascal VOC format.
    Does not apply complex augmentations, only loads the data.
    """
    def __init__(self, root, image_set='default', mode='RGB', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # Try 'JPEGImages' first (standard VOC), then 'Images'
        if os.path.exists(os.path.join(root, 'JPEGImages')):
            self.images_path = os.path.join(root, 'JPEGImages')
        else:
            self.images_path = os.path.join(root, 'Images')
            
        self.anno_path = os.path.join(root, 'Annotations')
        
        set_file = os.path.join(root, 'ImageSets/Main', f'{image_set}.txt')
        
        self.ids = []
        if os.path.exists(set_file):
            with open(set_file, 'r', encoding='utf-8-sig') as f:
                valid_ids = [line.strip() for line in f if line.strip()]
            
            for img_id in valid_ids:
                img_path = os.path.join(self.images_path, f'{img_id}.jpg')
                ann_path = os.path.join(self.anno_path, f'{img_id}.xml')
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    self.ids.append(img_id)
        
        print(f"Found {len(self.ids)} valid samples in {image_set} set (Root: {root}).")

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.images_path, f'{img_id}.jpg')
        ann_path = os.path.join(self.anno_path, f'{img_id}.xml')

        img = cv2.imread(img_path)
        if img is None: return None, None
        if self.mode == 'RGB': img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store image dimensions for albumentations
        h, w, _ = img.shape # Moved this line up

        tree = ET.parse(ann_path)
        boxes, labels = [], []
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text.strip()
            if name in class_to_idx:
                bbox = obj.find('bndbox')
                # Pascal VOC coordinates are 1-indexed, so subtract 1 for 0-indexed for albumentations
                xmin = float(bbox.find('xmin').text) - 1
                ymin = float(bbox.find('ymin').text) - 1
                xmax = float(bbox.find('xmax').text) - 1
                ymax = float(bbox.find('ymax').text) - 1
                
                # Ensure valid box (xmin < xmax, ymin < ymax)
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Clamp coordinates to image boundaries to prevent issues with albumentations internal checks
                xmin = max(0.0, min(xmin, w - 1))
                ymin = max(0.0, min(ymin, h - 1))
                xmax = max(0.0, min(xmax, w - 1))
                ymax = max(0.0, min(ymax, h - 1))
                
                # Re-check for valid box after clamping (it might become invalid if original box was very large)
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_to_idx[name])
        
        if not boxes: return None, None # This check should be after parsing all boxes

        target = {"boxes": np.array(boxes, dtype=np.float32), "labels": np.array(labels, dtype=np.int64)}
        
        if self.transform:
            transformed = self.transform(image=img, bboxes=target["boxes"], class_labels=target["labels"], height=h, width=w)
            img = transformed['image']
            target['boxes'] = transformed['bboxes']
            target['labels'] = transformed['class_labels']

        return img, target

    def __len__(self):
        return len(self.ids)

def get_voc_datasets(voc_root, img_size, years, transform_train=None, transform_val=None):
    """
    Helper function to get combined train and validation datasets for VOC.
    """
    train_datasets = []
    val_datasets = []
    
    # Check if root contains years directly
    # Original train.py defaults: voc_years=["2012"], voc_root="./data/VOC"
    for year in years:
        # Check standard VOC structure: root/VOC2012/Images etc.
        year_path = os.path.join(voc_root, f"VOC{year}")
        if not os.path.exists(year_path):
            # Try root/2012 if VOC prefix missing
            year_path = os.path.join(voc_root, year)
            if not os.path.exists(year_path):
                print(f"Warning: VOC {year} folder not found in {voc_root}")
                continue
        
        train_ds = PascalVOCDataset(year_path, image_set='trainval', transform=transform_train)
        val_ds = PascalVOCDataset(year_path, image_set='val', transform=transform_val)
        
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    if not train_datasets:
        # Fallback: maybe voc_root IS the dataset folder
        print(f"No year-specific folders found, trying {voc_root} directly...")
        train_ds = PascalVOCDataset(voc_root, image_set='trainval', transform=transform_train)
        val_ds = PascalVOCDataset(voc_root, image_set='val', transform=transform_val)
        if len(train_ds) > 0:
            print(f"Total: {len(train_ds)} train samples, {len(val_ds)} val samples.")
            return train_ds, val_ds
        else:
            raise FileNotFoundError(f"No VOC samples found in {voc_root}")

    train_combined = ConcatDataset(train_datasets)
    val_combined = ConcatDataset(val_datasets)
    print(f"\n--- Combined Dataset Statistics ---")
    print(f"Total Combined Train samples: {len(train_combined)}")
    print(f"Total Combined Val samples: {len(val_combined)}")
    print(f"-----------------------------------\n")

    return train_combined, val_combined

    def __len__(self):
        return len(self.ids)

def get_voc_datasets(voc_root, img_size, years, transform_train=None, transform_val=None):
    """
    Helper function to get combined train and validation datasets for VOC.
    """
    train_datasets = []
    val_datasets = []
    
    # Check if root contains years directly
    # Original train.py defaults: voc_years=["2012"], voc_root="./data/VOC"
    for year in years:
        # Check standard VOC structure: root/VOC2012/Images etc.
        year_path = os.path.join(voc_root, f"VOC{year}")
        if not os.path.exists(year_path):
            # Try root/2012 if VOC prefix missing
            year_path = os.path.join(voc_root, year)
            if not os.path.exists(year_path):
                print(f"Warning: VOC {year} folder not found in {voc_root}")
                continue
        
        train_ds = PascalVOCDataset(year_path, image_set='trainval', transform=transform_train)
        val_ds = PascalVOCDataset(year_path, image_set='val', transform=transform_val)
        
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    if not train_datasets:
        # Fallback: maybe voc_root IS the dataset folder
        print(f"No year-specific folders found, trying {voc_root} directly...")
        train_ds = PascalVOCDataset(voc_root, image_set='trainval', transform=transform_train)
        val_ds = PascalVOCDataset(voc_root, image_set='val', transform=transform_val)
        if len(train_ds) > 0:
            print(f"Total: {len(train_ds)} train samples, {len(val_ds)} val samples.")
            return train_ds, val_ds
        else:
            raise FileNotFoundError(f"No VOC samples found in {voc_root}")

    train_combined = ConcatDataset(train_datasets)
    val_combined = ConcatDataset(val_datasets)
    print(f"\n--- Combined Dataset Statistics ---")
    print(f"Total Combined Train samples: {len(train_combined)}")
    print(f"Total Combined Val samples: {len(val_combined)}")
    print(f"-----------------------------------\n")

    return train_combined, val_combined
