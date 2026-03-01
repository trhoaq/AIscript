import torch
from torch.utils.data import DataLoader
from data_loader import (
    get_base_transforms, 
    get_final_transforms, 
    get_eval_transforms, 
    safe_collate_fn
)
from dataset import MosaicMixupDataset, get_voc_datasets
from model.model import SSDGhost
from model.ssd_custom import SSDMobile
from trainer import DetectorTrainer
import json

def main():
    # 1. Load Config
    with open("config.json", "r") as f:
        config = json.load(f)

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
    model.to(device) # Move model to device BEFORE profiling

    # Calculate and log model complexity (total params and MAdds)
    try:
        from thop import profile, utils # Import utils
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        total_madds, total_params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"\n--- Model Complexity ---")
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Total MAdds (Giga): {total_madds / 1e9:.2f} G")
        print(f"------------------------\n")
        utils.remove_hooks(model) # Explicitly remove hooks after profiling
    except ImportError:
        print("Warning: 'thop' library not found. Skipping calculation of model parameters and MAdds.")
        print("Install with: pip install thop")
    except Exception as e:
        print(f"Warning: Could not calculate model complexity: {e}")


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
        'early_stopping_patience': config.get('early_stopping_patience', 10), # Pass patience
        'score_thresh': config.get('score_thresh', 0.05) # Pass score_thresh for post-processing
    }
    
    trainer = DetectorTrainer(model, train_loader, val_loader, device, trainer_config)

    # 6. Training Loop
    print(f"Starting training GhostNet SSD 0.5x on {device}...")
    for epoch in range(1, trainer_config['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        if train_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Train Loss: {train_loss:.4f}")
        
        # Evaluate after each epoch
        val_loss, val_mAP_0_5, val_precision, val_recall = trainer.evaluate_epoch(epoch)
        if val_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Val Loss: {val_loss:.4f} | mAP@0.5: {val_mAP_0_5:.4f} | P@0.5: {val_precision:.4f} | R@0.5: {val_recall:.4f}")

        # Early Stopping Logic
        if val_mAP_0_5 > trainer.best_val_map05:
            trainer.best_val_map05 = val_mAP_0_5
            trainer.epochs_no_improve = 0
            trainer.best_model_state = {
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'epoch': epoch,
                'val_mAP_0_5': val_mAP_0_5
            }
            if trainer.feature_adapters:
                trainer.best_model_state['adapters_state_dict'] = trainer.feature_adapters.state_dict()
            trainer.save_checkpoint("models/best_model.pth")
            print(f"Validation mAP@0.5 improved to {trainer.best_val_map05:.4f}. Saving best model state.")
        else:
            trainer.epochs_no_improve += 1
            print(f"Validation mAP@0.5 did not improve. Epochs without improvement: {trainer.epochs_no_improve}")

        if trainer.epochs_no_improve >= trainer.early_stopping_patience:
            print(f"Early stopping triggered after {trainer.early_stopping_patience} epochs without improvement.")
            if trainer.best_model_state:
                trainer.model.load_state_dict(trainer.best_model_state['model_state_dict'])
                trainer.optimizer.load_state_dict(trainer.best_model_state['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(trainer.best_model_state['scheduler_state_dict'])
                if trainer.feature_adapters and 'adapters_state_dict' in trainer.best_model_state:
                    trainer.feature_adapters.load_state_dict(trainer.best_model_state['adapters_state_dict'])
                print("Loaded best model weights from checkpoint.")
            break # Exit training loop

    # Save final model state after training (or early stopping)
    trainer.save_checkpoint(f"models/final_model.pth")

if __name__ == "__main__":
    main()
