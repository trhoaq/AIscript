import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from tqdm import tqdm
import os
from typing import List, Dict, Any
from model.utils import calculate_stats, compute_metrics # Import manual metrics
from wandb_utils import is_wandb_active, log_wandb
# Removed torchmetrics import

class DistillationLoss(nn.Module):
    """
    Distillation loss module.
    Combines KL-divergence for logits and MSE for feature maps.
    Assumes feature maps from teacher have been adapted to match student's channel count.
    """
    def __init__(self, kd_feature_weight=1.0, kd_logit_weight=1.0, temperature=2.0):
        super().__init__()
        self.kd_feature_weight = kd_feature_weight
        self.kd_logit_weight = kd_logit_weight
        self.temperature = temperature
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, student_feats, adapted_teacher_feats):
        # 1. Logit Distillation
        # WARNING: This simple truncation is problematic if anchor numbers differ significantly.
        if student_logits.shape != teacher_logits.shape:
            min_anchors = min(student_logits.size(1), teacher_logits.size(1))
            s_logits = student_logits[:, :min_anchors, :]
            t_logits = teacher_logits[:, :min_anchors, :]
        else:
            s_logits, t_logits = student_logits, teacher_logits

        soft_targets = F.softmax(t_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(s_logits / self.temperature, dim=-1)
        logit_loss = self.kl_loss(soft_prob, soft_targets) * (self.temperature ** 2)
        
        # 2. Feature Distillation
        feat_loss = 0
        for s_f, t_f in zip(student_feats, adapted_teacher_feats):
            feat_loss += self.mse_loss(s_f, t_f)
            
        total_loss = (self.kd_logit_weight * logit_loss) + (self.kd_feature_weight * feat_loss)
        return total_loss

class DetectorTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.device = device
        self.model = model.to(self.device) # Move main model to device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.use_wandb = is_wandb_active()

        self._maybe_load_pretrained_backbone(self.model)
        
        # Teacher model for Distillation
        self.teacher_model = config.get('teacher_model')
        self.distill_criterion = None
        self.feature_adapters = None
        
        params_to_optimize = list(self.model.parameters())

        if self.teacher_model:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            self._maybe_load_pretrained_backbone(self.teacher_model)

            self.distill_criterion = DistillationLoss(
                kd_feature_weight=config.get('kd_feature_weight', 1.0),
                kd_logit_weight=config.get('kd_logit_weight', 1.0)
            )
            student_feature_channels = getattr(self.model, "feature_channels", None)
            teacher_feature_channels = getattr(self.teacher_model, "feature_channels", None)
            if (
                isinstance(student_feature_channels, (list, tuple))
                and isinstance(teacher_feature_channels, (list, tuple))
                and len(student_feature_channels) > 0
                and len(teacher_feature_channels) > 0
            ):
                num_pairs = min(len(student_feature_channels), len(teacher_feature_channels))
                adapter_in_channels = [int(ch) for ch in teacher_feature_channels[:num_pairs]]
                adapter_out_channels = [int(ch) for ch in student_feature_channels[:num_pairs]]
            else:
                adapter_in_channels = [128, 128, 128, 128, 128]
                adapter_out_channels = [64, 64, 64, 64, 64]

            self.feature_adapters = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                    for in_ch, out_ch in zip(adapter_in_channels, adapter_out_channels)
                ]
            ).to(device)
            params_to_optimize.extend(self.feature_adapters.parameters())

        self.optimizer = optim.AdamW(
            params_to_optimize, 
            lr=config.get('lr', 1e-3), 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 100))
        
        # --- Metrics and Early Stopping ---
        self.score_thresh = config.get('score_thresh', 0.05) # From config.json
        self.eval_score_thresh = config.get('eval_score_thresh', self.score_thresh)
        self.eval_pre_nms_topk = config.get('eval_pre_nms_topk', 400)
        self.eval_max_detections = config.get('eval_max_detections', 100)
        
        self.best_val_map05 = -1.0 # Track best mAP@0.5 for early stopping
        self.epochs_no_improve = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 10) # Default patience
        self.best_model_state: Dict[str, Any] = {} # To store best model weights and adapter weights
        
        # Interval tracking (for saving every 10 epochs)
        self.interval_best_map05 = -1.0
        self.interval_best_state: Dict[str, Any] = {}
        
        print(f"Early stopping patience set to: {self.early_stopping_patience}")

    def _maybe_load_pretrained_backbone(self, model: nn.Module) -> None:
        """Load ImageNet pretrained backbone if model exposes the SSD pretrained API."""
        has_api = (
            hasattr(model, "pretrained_backbone")
            and hasattr(model, "backbone_has_weights_loaded")
            and hasattr(model, "load_pretrained_weights")
        )
        if not has_api:
            return

        if bool(getattr(model, "pretrained_backbone")) and not bool(getattr(model, "backbone_has_weights_loaded")):
            model.load_pretrained_weights(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        if self.feature_adapters:
            self.feature_adapters.train()

        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            if batch[0] is None:
                continue
            images, targets = batch
            
            images = torch.stack([img.to(self.device) for img in images])
            # Ensure targets are tensors and on correct device
            targets = [{k: torch.as_tensor(v).to(self.device) for k, v in t.items()} for t in targets]

            # Forward Student
            self.optimizer.zero_grad()
            
            student_logits, student_regs, student_feats = self.model.forward_logits(images)
            
            if hasattr(self.model, "generate_priors"):
                priors = self.model.generate_priors(student_feats, images)
            else:
                feat_sizes = [(f.size(2), f.size(3)) for f in student_feats]
                priors = self.model.anchor_generator.generate(feat_sizes, self.model.img_size, student_logits.device)
            gt_loss_dict = self.model.multibox_loss(student_logits, student_regs, targets, priors)
            
            # Ground truth loss
            loss = gt_loss_dict["bbox_regression"] + gt_loss_dict["classification"]

            # Knowledge Distillation Loss
            if self.teacher_model and self.distill_criterion and self.feature_adapters:
                with torch.no_grad():
                    teacher_logits, _, teacher_feats_raw = self.teacher_model.forward_logits(images)
                
                # Adapt teacher features
                adapted_teacher_feats = []
                num_feature_pairs = min(len(student_feats), len(teacher_feats_raw), len(self.feature_adapters))
                for i in range(num_feature_pairs):
                    teacher_feature = self.feature_adapters[i](teacher_feats_raw[i])
                    if teacher_feature.shape[-2:] != student_feats[i].shape[-2:]:
                        teacher_feature = F.interpolate(
                            teacher_feature,
                            size=student_feats[i].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    adapted_teacher_feats.append(teacher_feature)

                kd_loss = self.distill_criterion(
                    student_logits,
                    teacher_logits,
                    student_feats[:num_feature_pairs],
                    adapted_teacher_feats,
                )
                loss = loss + kd_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        if self.use_wandb:
            log_wandb(
                {
                    "train/loss": avg_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
            
        self.scheduler.step()
        return avg_loss

    def evaluate_epoch(self, epoch):
        if self.val_loader is None:
            print("Validation loader not provided. Skipping evaluation.")
            return None, None, None, None, None, None, None

        self.model.eval()
        if self.feature_adapters:
            self.feature_adapters.eval()

        total_val_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        # For mAP calculation
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if batch[0] is None:
                    # print(f"Warning: Batch {batch_idx} in val_loader contained None values. Skipping.")
                    continue
                images, targets = batch
                
                images = torch.stack([img.to(self.device) for img in images])
                # Ensure targets are tensors and on correct device
                original_targets = [{k: torch.as_tensor(v).to(self.device) for k, v in t.items()} for t in targets] # Ground Truth

                student_logits, student_regs, student_feats = self.model.forward_logits(images)
                
                if hasattr(self.model, "generate_priors"):
                    priors = self.model.generate_priors(student_feats, images)
                else:
                    feat_sizes = [(f.size(2), f.size(3)) for f in student_feats]
                    priors = self.model.anchor_generator.generate(feat_sizes, self.model.img_size, student_logits.device)
                
                gt_loss_dict = self.model.multibox_loss(student_logits, student_regs, original_targets, priors)
                
                loss = gt_loss_dict["bbox_regression"] + gt_loss_dict["classification"]

                total_val_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())

                # Post-process model outputs for mAP calculation
                detections = self.model.post_process(
                    student_logits,
                    student_regs,
                    priors,
                    self.model.img_size,
                    self.eval_score_thresh,
                    pre_nms_topk=self.eval_pre_nms_topk,
                    max_detections=self.eval_max_detections,
                )
                
                for i in range(len(detections)):
                    # Keep on original device (GPU) to use GPU acceleration for IoU later
                    all_preds.append(detections[i]) 
                    all_targets.append({
                        "boxes": original_targets[i]["boxes"],
                        "labels": original_targets[i]["labels"],
                    })
        
        # Compute mAP, Precision, Recall at IoU 0.5 and 0.95 using manual utilities
        avg_val_loss = total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

        if len(all_preds) > 0 and len(all_targets) > 0:
            stats_0_5 = calculate_stats(all_preds, all_targets, iou_threshold=0.5)
            mAP_0_5, precision_0_5, recall_0_5 = compute_metrics(stats_0_5)
            stats_0_95 = calculate_stats(all_preds, all_targets, iou_threshold=0.95)
            mAP_0_95, precision_0_95, recall_0_95 = compute_metrics(stats_0_95)
        else:
            mAP_0_5, precision_0_5, recall_0_5 = 0.0, 0.0, 0.0
            mAP_0_95, precision_0_95, recall_0_95 = 0.0, 0.0, 0.0
            print("Warning: No valid predictions or targets for metric calculation in this epoch. Setting metrics to 0.")
        f1_0_5 = 2 * precision_0_5 * recall_0_5 / (precision_0_5 + recall_0_5 + 1e-16)
        f1_0_95 = 2 * precision_0_95 * recall_0_95 / (precision_0_95 + recall_0_95 + 1e-16)

        # Return relevant metrics
        if self.use_wandb:
            log_wandb(
                {
                    "val/loss": avg_val_loss,
                    "val/mAP@0.5": mAP_0_5,
                    "val/Precision@0.5": precision_0_5,
                    "val/Recall@0.5": recall_0_5,
                    "val/F1@0.5": f1_0_5,
                    "val/mAP@0.95": mAP_0_95,
                    "val/Precision@0.95": precision_0_95,
                    "val/Recall@0.95": recall_0_95,
                    "val/F1@0.95": f1_0_95,
                },
                step=epoch,
            )

        # Track interval best
        if mAP_0_5 > self.interval_best_map05:
            self.interval_best_map05 = mAP_0_5
            self.interval_best_state = {
                'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': {k: v for k, v in self.optimizer.state_dict().items()}, # Simplified
                'scheduler_state_dict': {k: v for k, v in self.scheduler.state_dict().items()},
                'best_val_map05': self.interval_best_map05,
                'epoch': epoch
            }
            if self.feature_adapters:
                self.interval_best_state['adapters_state_dict'] = {k: v.cpu().clone() for k, v in self.feature_adapters.state_dict().items()}

        # Free temporary evaluation buffers between epochs.
        all_preds.clear()
        all_targets.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_val_loss, mAP_0_5, precision_0_5, recall_0_5, mAP_0_95, precision_0_95, recall_0_95

    def save_interval_checkpoints(self, epoch):
        """Saves 'last' and 'best' models for the current 10-epoch interval."""
        base_dir = self.config.get("checkpoint_dir", "models")
        
        # Save 'last'
        last_path = os.path.join(base_dir, f"epoch_{epoch}_last.pth") if base_dir else f"epoch_{epoch}_last.pth"
        self.save_checkpoint(last_path, epoch=epoch)
        
        # Save 'best' of this interval
        if self.interval_best_state:
            best_path = os.path.join(base_dir, f"epoch_{epoch}_best.pth") if base_dir else f"epoch_{epoch}_best.pth"
            torch.save(self.interval_best_state, best_path)
            print(f"Saved interval best checkpoint to {best_path} (mAP@0.5: {self.interval_best_map05:.4f})")
        
        # Reset interval tracking
        self.interval_best_map05 = -1.0
        self.interval_best_state = {}

    def save_checkpoint(self, path, epoch=None):
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), # Added scheduler
            'best_val_map05': self.best_val_map05, # Added best metric
            'epoch': epoch # Added epoch
        }
        if self.feature_adapters:
            state['adapters_state_dict'] = self.feature_adapters.state_dict()

        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"Warning: Checkpoint file {path} not found. Starting from scratch.")
            return 0

        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_map05 = checkpoint.get('best_val_map05', -1.0)
        
        if self.feature_adapters and 'adapters_state_dict' in checkpoint:
            self.feature_adapters.load_state_dict(checkpoint['adapters_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}.")
        return start_epoch
