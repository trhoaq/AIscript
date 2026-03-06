import torch # type: ignore
import os, sys
import gc
import re
from torch.utils.data import DataLoader # type: ignore
from config_utils import load_merged_config
from data_loader import (
    get_base_transforms, 
    get_final_transforms, 
    get_eval_transforms, 
    safe_collate_fn
)
from dataset import MosaicMixupDataset, get_coco_datasets, get_voc_datasets, set_seed_everything
# ONLY IMPORT SSDMobile, this is the model we are training (the teacher)
from model.ssdlite_mobilenet import SSDMobile 
from trainer import DetectorTrainer
from wandb_utils import finish_wandb, init_wandb, log_wandb
from typing import Dict, Any, Optional


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.lower().endswith(".pth")
    ]
    if not checkpoints:
        return None

    def sort_key(path: str):
        name = os.path.basename(path)
        match = re.match(r"epoch_(\d+)_(last|best)\.pth$", name)
        if match:
            epoch = int(match.group(1))
            kind_priority = 2 if match.group(2) == "last" else 1
            return (1, epoch, kind_priority, os.path.getmtime(path))
        return (0, 0, 0, os.path.getmtime(path))

    return max(checkpoints, key=sort_key)

def main():
    # 1. Load Config
    try:
        config = load_merged_config("config/config.json")
    except FileNotFoundError:
        print("config.json not found, using default settings.")
        sys.exit()
        

    init_wandb(config, default_run_name="Teacher-Training")

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_seed_everything(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config.get("img_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    dataset_format = str(config.get("dataset_format", "voc")).strip().lower()
    voc_years = config.get("voc_years", ["2012"])
    voc_root = config.get("voc_root", "./data/VOC")
    voc_auto_split = bool(config.get("voc_auto_split", True))
    voc_val_ratio = float(config.get("voc_val_ratio", 0.2))
    coco_root = config.get("coco_root", "./data")
    coco_train_split = config.get("coco_train_split", config.get("train_split", "train2017"))
    coco_val_split = config.get("coco_val_split", config.get("val_split", "val2017"))
    teacher_best_model_path = config.get("teacher_best_model_path", "models/teacher_best_model.pth")
    teacher_final_model_path = config.get("teacher_final_model_path", "models/teacher_final_model.pth")
    teacher_interval_checkpoint_dir = config.get("teacher_interval_checkpoint_dir", "models")
    eval_interval = max(1, int(config.get("eval_interval", 1)))

    # 2. Dataset & Loader
    if dataset_format == "voc":
        base_train_ds, base_val_ds = get_voc_datasets(
            voc_root=voc_root,
            img_size=img_size,
            years=voc_years,
            transform_train=get_base_transforms(img_size),
            transform_val=get_eval_transforms(img_size),
            train_split=config.get("train_split", "trainval"),
            val_split=config.get("val_split", "val"),
            obj_classes=config.get("obj_classes"),
            auto_split=voc_auto_split,
            split_seed=seed,
            val_ratio=voc_val_ratio,
        )
    elif dataset_format == "coco":
        base_train_ds, base_val_ds = get_coco_datasets(
            coco_root=coco_root,
            img_size=img_size,
            transform_train=get_base_transforms(img_size),
            transform_val=get_eval_transforms(img_size),
            train_split=coco_train_split,
            val_split=coco_val_split,
            obj_classes=config.get("obj_classes"),
        )
    else:
        raise ValueError("dataset_format must be either 'voc' or 'coco'.")

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
    use_pretrained_teacher_backbone = True
    teacher_pretrained_backbone_model = "mobilenetv3_large_100"
    model = SSDMobile(
        num_classes=num_classes,
        aspect_ratios=teacher_aspect_ratios,
        img_size=img_size,
        s_min=0.07,
        s_max=0.95,
        pretrained_backbone=use_pretrained_teacher_backbone, # Pass the new parameter
        pretrained_backbone_model_name=teacher_pretrained_backbone_model,
    )
    model.to(device) # Move model to device BEFORE profiling

    # Calculate and log model complexity (total params and MAdds)
    try:
        import copy
        from thop import profile
        # Create a dummy copy to avoid polluting the main model with hooks
        model_copy = copy.deepcopy(model).to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            total_madds, total_params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        print(f"\n--- Teacher Model Complexity ---")
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Total MAdds (Giga): {total_madds / 1e9:.2f} G")
        print(f"--------------------------------\n")
        del model_copy, dummy_input # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        'checkpoint_dir': teacher_interval_checkpoint_dir,
    }
    
    # Pass None for teacher_model as we are training the model itself
    trainer = DetectorTrainer(model, train_loader, val_loader, device, trainer_config)

    # Resume training if checkpoint exists
    resume_checkpoint = config.get("resume_checkpoint")
    start_epoch = 0
    resume_path = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        resume_path = resume_checkpoint
    elif resume_checkpoint:
        print(f"Configured resume checkpoint not found: {resume_checkpoint}. Falling back to latest checkpoint in dir.")

    if resume_path is None:
        latest_checkpoint = find_latest_checkpoint(teacher_interval_checkpoint_dir)
        if latest_checkpoint:
            resume_path = latest_checkpoint
            print(f"Auto-resume found latest teacher checkpoint: {resume_path}")

    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)
    else:
        print("No teacher checkpoint found. Starting from scratch.")

    # 5. Training Loop
    print(f"Starting training SSDMobile Teacher Model on {device}...")
    for epoch in range(start_epoch + 1, trainer_config['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        if train_loss is not None:
            print(f"Epoch {epoch}/{trainer_config['epochs']} | Train Loss: {train_loss:.4f}")
        
        # Evaluate based on configured interval to avoid expensive validation every epoch.
        should_eval = (epoch % eval_interval == 0)
        if should_eval:
            (
                val_loss,
                val_mAP_0_5,
                val_precision_0_5,
                val_recall_0_5,
                val_mAP_0_95,
                val_precision_0_95,
                val_recall_0_95,
            ) = trainer.evaluate_epoch(epoch)
            if val_loss is not None:
                print(
                    f"Epoch {epoch}/{trainer_config['epochs']} | Val Loss: {val_loss:.4f} "
                    f"| mAP@0.5: {val_mAP_0_5:.4f} | P@0.5: {val_precision_0_5:.4f} | R@0.5: {val_recall_0_5:.4f} "
                    f"| mAP@0.95: {val_mAP_0_95:.4f} | P@0.95: {val_precision_0_95:.4f} | R@0.95: {val_recall_0_95:.4f}"
                )

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
                trainer.save_checkpoint(teacher_best_model_path)
                print(f"Validation mAP@0.5 improved to {trainer.best_val_map05:.4f}. Saving best teacher model state.")
            else:
                trainer.epochs_no_improve += 1
                print(f"Validation mAP@0.5 did not improve. Epochs without improvement: {trainer.epochs_no_improve}")

            log_wandb(
                {
                    "val/best_mAP@0.5": trainer.best_val_map05,
                    "train/epochs_no_improve": trainer.epochs_no_improve,
                },
                step=epoch,
            )
        else:
            print(f"Skipping validation at epoch {epoch} (eval_interval={eval_interval}).")
            log_wandb({"val/skipped": 1}, step=epoch)

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
    trainer.save_checkpoint(teacher_final_model_path)

    finish_wandb()

if __name__ == "__main__":
    main()
