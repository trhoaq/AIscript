import torch
import os
import gc
from typing import Any, Dict
from torch.utils.data import DataLoader
from config_utils import load_merged_config
from data_loader import (
    get_base_transforms, 
    get_final_transforms, 
    get_eval_transforms, 
    safe_collate_fn
)
from dataset import MosaicMixupDataset, get_coco_datasets, get_voc_datasets, set_seed_everything
from model.model import SSDGhost
from model.ssdlite_mobilenet import SSDMobile
from trainer import DetectorTrainer
from wandb_utils import finish_wandb, init_wandb, log_wandb

def load_teacher_weights(teacher: SSDMobile, checkpoint_path: str, device: torch.device) -> None:
    """
    Load teacher checkpoint weights for KD.
    Supports both full checkpoint dicts and raw model state_dict files.
    """
    if not checkpoint_path:
        raise ValueError("teacher_checkpoint is empty. Please provide a valid checkpoint path.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    print(f"Loading teacher checkpoint from: {checkpoint_path}")
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    try:
        teacher.load_state_dict(state_dict, strict=True)
        print("Teacher checkpoint loaded with strict=True.")
    except RuntimeError as exc:
        print(f"Warning: strict load failed for teacher checkpoint ({exc}). Retrying with strict=False.")
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        print(f"Teacher checkpoint loaded with strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    # Prevent trainer from reloading backbone pretrained weights over checkpoint values.
    if hasattr(teacher, "backbone_has_weights_loaded"):
        teacher.backbone_has_weights_loaded = True

def main():
    # 1. Load Config
    config = load_merged_config("config/config.json")

    init_wandb(config)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_seed_everything(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config.get("img_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    student_width = float(config.get("student_width", 1.0))
    use_pretrained_student_backbone = bool(
        config.get("pretrain", config.get("use_pretrained_student_backbone", True))
    )
    student_pretrained_backbone_model = str(
        config.get(
            "student_pretrained_backbone_model",
            config.get("student_imagenet_pretrained_model", "mobilenetv3_large_100"),
        )
    ).strip()
    load_student_checkpoint_enabled = bool(config.get("load_student_checkpoint", False))
    student_checkpoint = config.get("student_checkpoint", "models/student_checkpoint.pth")
    student_best_model_path = config.get("student_best_model_path", "models/best_model.pth")
    student_final_model_path = config.get("student_final_model_path", "models/final_model.pth")
    student_interval_checkpoint_dir = config.get("student_interval_checkpoint_dir", "models")
    dataset_format = str(config.get("dataset_format", "voc")).strip().lower()
    voc_years = config.get("voc_years", ["2012"])
    voc_root = config.get("voc_root", "./data/VOC")
    voc_auto_split = bool(config.get("voc_auto_split", True))
    voc_val_ratio = float(config.get("voc_val_ratio", 0.2))
    coco_root = config.get("coco_root", "./data")
    coco_train_split = config.get("coco_train_split", config.get("train_split", "train2017"))
    coco_val_split = config.get("coco_val_split", config.get("val_split", "val2017"))
    eval_interval = max(1, int(config.get("eval_interval", 1)))
    use_kd = bool(config.get("use_kd", True))
    load_teacher_checkpoint_enabled = bool(config.get("load_teacher_checkpoint", True))

    # 2. Dataset & Loader
    if dataset_format == "voc":
        # Get combined train and validation datasets using VOC helper.
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
        # COCO loader auto-detects directories with or without "_coco" suffix.
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

    # 3. Model
    num_classes = len(config["obj_classes"])
    model = SSDGhost(
        num_classes=num_classes,
        width=student_width,
        img_size=img_size,
        pretrained_backbone=use_pretrained_student_backbone,
        pretrained_backbone_model_name=student_pretrained_backbone_model,
    )
    model.to(device) # Move model to device BEFORE profiling
    if use_pretrained_student_backbone:
        model.load_pretrained_weights(device)

    # Calculate and log model complexity (total params and MAdds)
    try:
        import copy
        from thop import profile
        # Create a dummy copy to avoid polluting the main model with hooks
        model_copy = copy.deepcopy(model).to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            total_madds, total_params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        print(f"\n--- Model Complexity ---")
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Total MAdds (Giga): {total_madds / 1e9:.2f} G")
        print(f"------------------------\n")
        del model_copy, dummy_input # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        print("Warning: 'thop' library not found. Skipping calculation of model parameters and MAdds.")
        print("Install with: pip install thop")
    except Exception as e:
        print(f"Warning: Could not calculate model complexity: {e}")


    # 4. Teacher Model (SSDLite MobileNetV3 from ssdlite_mobilenet.py)
    teacher = None
    if use_kd:
        teacher_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        teacher_checkpoint = config.get("teacher_checkpoint", "models/teacher_best_model.pth")
        use_pretrained_teacher_backbone = config.get("use_pretrained_teacher_backbone", True)
        teacher_pretrained_backbone_model = str(
            config.get("teacher_pretrained_backbone_model", "mobilenetv3_large_100")
        ).strip()
        teacher = SSDMobile(
            num_classes=num_classes,
            aspect_ratios=teacher_aspect_ratios,
            img_size=img_size,
            s_min=0.07,
            s_max=0.95,
            pretrained_backbone=use_pretrained_teacher_backbone,
            pretrained_backbone_model_name=teacher_pretrained_backbone_model,
        )

        if load_teacher_checkpoint_enabled:
            load_teacher_weights(teacher, teacher_checkpoint, device)
        else:
            print("Skipping teacher checkpoint loading (load_teacher_checkpoint=False).")

        # For distillation, teacher should be in eval mode
        teacher.eval()
    else:
        print("Knowledge distillation disabled (use_kd=False). Training without teacher model.")

    # 5. Trainer
    trainer_config = {
        'lr': config.get('lr', 1e-3),
        'epochs': config.get('epochs', 100),
        'teacher_model': teacher,
        'weight_decay': config.get('weight_decay', 1e-4),
        'kd_feature_weight': config.get('kd_feature_weight', 1.0),
        'kd_logit_weight': config.get('kd_logit_weight', 1.0),
        'early_stopping_patience': config.get('early_stopping_patience', 10), # Pass patience
        'score_thresh': config.get('score_thresh', 0.05), # Pass score_thresh for post-processing
        'eval_score_thresh': config.get('eval_score_thresh', config.get('score_thresh', 0.05)),
        'eval_pre_nms_topk': config.get('eval_pre_nms_topk', 400),
        'eval_max_detections': config.get('eval_max_detections', 100),
        'checkpoint_dir': student_interval_checkpoint_dir,
    }
    
    trainer = DetectorTrainer(model, train_loader, val_loader, device, trainer_config)

    # Resume training if checkpoint exists
    start_epoch = 0
    if load_student_checkpoint_enabled:
        if not student_checkpoint:
            raise ValueError("student_checkpoint is empty. Please provide a valid checkpoint path.")
        if not os.path.exists(student_checkpoint):
            raise FileNotFoundError(f"Student checkpoint not found: {student_checkpoint}")
        start_epoch = trainer.load_checkpoint(student_checkpoint)
    else:
        resume_checkpoint = config.get("resume_checkpoint")
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            start_epoch = trainer.load_checkpoint(resume_checkpoint)

    # 6. Training Loop
    print(f"Starting training SSDLite MobileNetV3 {student_width:.1f}x on {device}...")
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
                if trainer.feature_adapters:
                    trainer.best_model_state['adapters_state_dict'] = trainer.feature_adapters.state_dict()
                trainer.save_checkpoint(student_best_model_path)
                print(f"Validation mAP@0.5 improved to {trainer.best_val_map05:.4f}. Saving best model state.")
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
    trainer.save_checkpoint(student_final_model_path)
    
    finish_wandb()

if __name__ == "__main__":
    main()
