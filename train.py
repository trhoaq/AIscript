import torch
from torch.utils.data import DataLoader
from data_loader import (
    MosaicMixupDataset,
    get_base_transforms, 
    get_final_transforms, 
    get_eval_transforms, 
    safe_collate_fn
)
from dataset.voc import get_voc_datasets # Import the new VOC dataset helper
from model.model import SSDGhost
from model.ssd_custom import SSDMobile
from trainer import DetectorTrainer
import json

def main():
    # 1. Load Config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("config.json not found, using default settings.")
        config = {
            "obj_classes": ["bg", "cat", "dog"], 
            "lr": 1e-3, 
            "epochs": 50, 
            "p_mosaic": 0.5, 
            "p_mixup": 0.5,
            "img_size": 256,
            "batch_size": 32,
            "num_workers": 4,
            "voc_years": ["2012"], # Default to VOC2012
            "voc_root": "./data/VOC"
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config.get("img_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    voc_years = config.get("voc_years", ["2012"])
    voc_root = config.get("voc_root", "./data/VOC")

    # 2. Dataset & Loader
    # Get combined train and validation datasets using the new helper
    base_train_ds, base_val_ds = get_voc_datasets(
        voc_root=voc_root,
        img_size=img_size,
        years=voc_years,
        transform_train=get_base_transforms(img_size), # Base transforms for individual images
        transform_val=get_eval_transforms(img_size) # Eval transforms for validation
    )

    # Wrap training dataset with Mosaic and MixUp
    train_ds = MosaicMixupDataset(
        base_dataset=base_train_ds,
        img_size=img_size,
        p_mosaic=config.get("p_mosaic", 0.5),
        p_mixup=config.get("p_mixup", 0.5),
        transform=get_final_transforms(img_size), # Final transforms after Mosaic/MixUp
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=safe_collate_fn
    )

    val_loader = DataLoader(
        base_val_ds, # Validation set does not use Mosaic/MixUp
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=safe_collate_fn
    )

    # 3. Model
    num_classes = len(config["obj_classes"])
    model = SSDGhost(num_classes=num_classes, width=0.5, img_size=img_size)

    # 4. Teacher Model (MobileNetV3 SSD from ssd_custom.py)
    teacher_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    teacher = SSDMobile(
        num_classes=num_classes,
        aspect_ratios=teacher_aspect_ratios,
        img_size=img_size,
        s_min=0.07,
        s_max=0.95,
    )
    # For distillation, teacher should be in eval mode
    teacher.eval()

    # 5. Trainer
    trainer_config = {
        'lr': config.get('lr', 1e-3),
        'epochs': config.get('epochs', 100),
        'teacher_model': teacher,
        'weight_decay': config.get('weight_decay', 1e-4),
        'kd_feature_weight': config.get('kd_feature_weight', 1.0),
        'kd_logit_weight': config.get('kd_logit_weight', 1.0),
    }
    
    trainer = DetectorTrainer(model, train_loader, val_loader, device, trainer_config)

    # 6. Training Loop
    print(f"Starting training GhostNet SSD 0.5x on {device}...")
    for epoch in range(1, trainer_config['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        if train_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Train Loss: {train_loss:.4f}")
        
        val_loss = trainer.evaluate_epoch(epoch)
        if val_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Val Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            trainer.save_checkpoint(f"models/ssdghost_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
