import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore
import cv2 # pyright: ignore[reportMissingImports]

# --- Augmentation Pipelines ---

def get_base_transforms(img_size=256):
    """Transforms for the base dataset, applying fixed padding then resizing."""
    return A.Compose([
        A.PadIfNeeded(min_height=1000, min_width=1000, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        A.Resize(height=img_size, width=img_size, p=1.0), # Resize padded square image to img_size x img_size
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

def get_final_transforms(img_size=256):
    """Transforms to apply after Mosaic/MixUp or on single images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, 
                 translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
                 rotate=(-10, 10), 
                 fill_value=0, 
                 interpolation=cv2.INTER_LINEAR, 
                 border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 1), 
                        hole_height_range=(int(img_size // 8), int(img_size // 8)), 
                        hole_width_range=(int(img_size // 8), int(img_size // 8)), 
                        p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

# Alias for backward compatibility
get_train_transforms = get_final_transforms

def get_eval_transforms(img_size=256):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))


def safe_collate_fn(batch):
    """
    A collate function that filters out None values from the batch.
    """
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None, None
    return tuple(zip(*batch))
