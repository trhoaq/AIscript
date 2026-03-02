import torch
import os
from torch.utils.data import DataLoader
from data_loader import (
    get_base_transforms, 
    get_final_transforms, 
    get_eval_transforms, 
    safe_collate_fn
)
from dataset import MosaicMixupDataset, get_voc_datasets
# ONLY IMPORT SSDMobile, this is the model we are training (the teacher)
from model.ssd_custom import SSDMobile 
from trainer import DetectorTrainer
import json
from typing import Dict, Any

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

    # Initialize wandb
    try:
        import wandb
        if config.get("wandb_project"):
            wandb.init(
                project=config["wandb_project"],
                name=config.get("wandb_run_name", "Teacher-Training"),
                config=config
            )
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb initialization.")
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config.get("img_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    voc_years = config.get("voc_years", ["2012"])
    voc_root = config.get("voc_root", "./data/VOC")
    eval_interval = max(1, int(config.get("eval_interval", 1)))

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

    # 3. Model (This will be the SSDMobile model)
    num_classes = len(config["obj_classes"])
    teacher_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]] # SSDMobile's aspect ratios
    use_pretrained_teacher_backbone = config.get("use_pretrained_teacher_backbone", True) # Default to True
    model = SSDMobile(
        num_classes=num_classes,
        aspect_ratios=teacher_aspect_ratios,
        img_size=img_size,
        s_min=0.07,
        s_max=0.95,
        pretrained_backbone=use_pretrained_teacher_backbone, # Pass the new parameter
    )
    model.to(device) # Move model to device BEFORE profiling

    # Calculate and log model complexity (total params and MAdds)
    try:
        import copy
        from thop import profile
        # Create a dummy copy to avoid polluting the main model with hooks
        model_copy = copy.deepcopy(model).to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        total_madds, total_params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        print(f"\n--- Teacher Model Complexity ---")
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Total MAdds (Giga): {total_madds / 1e9:.2f} G")
        print(f"--------------------------------\n")
        del model_copy # Cleanup
    except ImportError:
        print("Warning: 'thop' library not found. Skipping calculation of teacher model parameters and MAdds.")
        print("Install with: pip install thop")
    except Exception as e:
        print(f"Warning: Could not calculate teacher model complexity: {e}")

    # 4. Trainer
    trainer_config: Dict[str, Any] = { # Add type hint for trainer_config
        'lr': config.get('lr', 1e-3),
        'epochs': config.get('epochs', 100),
        # No teacher_model, kd_weights needed for training the teacher itself
        'weight_decay': config.get('weight_decay', 1e-4),
        'early_stopping_patience': config.get('early_stopping_patience', 10), # Pass patience
        'score_thresh': config.get('score_thresh', 0.05), # Pass score_thresh for post-processing
        'eval_score_thresh': config.get('eval_score_thresh', config.get('score_thresh', 0.05)),
        'eval_pre_nms_topk': config.get('eval_pre_nms_topk', 400),
        'eval_max_detections': config.get('eval_max_detections', 100),
    }
    
    # Pass None for teacher_model as we are training the model itself
    trainer = DetectorTrainer(model, train_loader, val_loader, device, trainer_config)

    # Resume training if checkpoint exists
    resume_checkpoint = config.get("resume_checkpoint")
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch = trainer.load_checkpoint(resume_checkpoint)

    # 5. Training Loop
    print(f"Starting training SSDMobile Teacher Model on {device}...")
    for epoch in range(start_epoch + 1, trainer_config['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        if train_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Train Loss: {train_loss:.4f}")
        
        # Evaluate based on configured interval to avoid expensive validation every epoch.
        should_eval = (epoch % eval_interval == 0)
        if should_eval:
            val_loss, val_mAP_0_5, val_precision, val_recall = trainer.evaluate_epoch(epoch)
            if val_loss is not None:
                print(f"Epoch {epoch}/{trainer_config['epochs']} | Val Loss: {val_loss:.4f} | mAP@0.5: {val_mAP_0_5:.4f} | P@0.5: {val_precision:.4f} | R@0.5: {val_recall:.4f}")

            # Early Stopping Logic (only update when an eval is run)
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
                # No feature_adapters for teacher model
                trainer.save_checkpoint("models/teacher_best_model.pth")
                print(f"Validation mAP@0.5 improved to {trainer.best_val_map05:.4f}. Saving best teacher model state.")
            else:
                trainer.epochs_no_improve += 1
                print(f"Validation mAP@0.5 did not improve. Epochs without improvement: {trainer.epochs_no_improve}")
        else:
            print(f"Skipping validation at epoch {epoch} (eval_interval={eval_interval}).")

        # Periodic checkpoint saving (every 10 epochs)
        if epoch % 10 == 0:
            trainer.save_interval_checkpoints(epoch)

        if trainer.epochs_no_improve >= trainer.early_stopping_patience:
            print(f"Early stopping triggered for Teacher Model after {trainer.early_stopping_patience} epochs without improvement.")
            if trainer.best_model_state:
                trainer.model.load_state_dict(trainer.best_model_state['model_state_dict'])
                trainer.optimizer.load_state_dict(trainer.best_model_state['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(trainer.best_model_state['scheduler_state_dict'])
                # No feature_adapters for teacher model
                print("Loaded best teacher model weights from checkpoint.")
            break # Exit training loop

    # Save final model state after training (or early stopping)
    trainer.save_checkpoint(f"models/teacher_final_model.pth")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
