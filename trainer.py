import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from typing import List, Dict, Any
from torchmetrics.detection import MeanAveragePrecision # Import mAP metric
from model.ssd_custom import SSDMobile
# You might need to install torchmetrics: pip install torchmetrics torchvision

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

        # If the main model is SSDMobile and pretrained_backbone is enabled, load its pretrained weights
        from model.ssd_custom import SSDMobile
        if isinstance(self.model, SSDMobile) and self.model.pretrained_backbone and not self.model.backbone_has_weights_loaded:
            self.model.load_pretrained_weights(self.device)
        
        # Teacher model for Distillation
        self.teacher_model = config.get('teacher_model')
        self.distill_criterion = None
        self.feature_adapters = None
        
        params_to_optimize = list(self.model.parameters())

        if self.teacher_model:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            
            # If the teacher model is SSDMobile, ensure its weights are loaded
            if isinstance(self.teacher_model, SSDMobile) and self.teacher_model.pretrained_backbone and not self.teacher_model.backbone_has_weights_loaded:
                self.teacher_model.load_pretrained_weights(self.device)

            self.distill_criterion = DistillationLoss(
                kd_feature_weight=config.get('kd_feature_weight', 1.0),
                kd_logit_weight=config.get('kd_logit_weight', 1.0)
            )
            # Adapters to match teacher's 128 channels to student's 64 channels
            # We will match the first 5 feature maps
            self.feature_adapters = nn.ModuleList([
                nn.Conv2d(128, 64, kernel_size=1) for _ in range(5)
            ]).to(device)
            params_to_optimize.extend(self.feature_adapters.parameters())

        self.optimizer = optim.AdamW(
            params_to_optimize, 
            lr=config.get('lr', 1e-3), 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 100))
        
        # --- Metrics and Early Stopping ---
        self.map_metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False).to(device) # Only overall mAP@0.5
        self.score_thresh = config.get('score_thresh', 0.05) # From config.json
        
        self.best_val_map05 = -1.0 # Track best mAP@0.5 for early stopping
        self.epochs_no_improve = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 10) # Default patience
        self.best_model_state: Dict[str, Any] = {} # To store best model weights and adapter weights
        print(f"Early stopping patience set to: {self.early_stopping_patience}")

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
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward Student
            self.optimizer.zero_grad()
            
            student_logits, student_regs, student_feats = self.model.forward_logits(images)
            
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
                for i in range(len(student_feats)): # Should be 5
                    adapted_teacher_feats.append(self.feature_adapters[i](teacher_feats_raw[i]))

                kd_loss = self.distill_criterion(student_logits, teacher_logits, student_feats, adapted_teacher_feats)
                loss = loss + kd_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        self.scheduler.step()
        return total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0

    def evaluate_epoch(self, epoch):
        if self.val_loader is None:
            print("Validation loader not provided. Skipping evaluation.")
            return None, None, None # Return three None values for mAP, P, R

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
                original_targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] # Ground Truth

                student_logits, student_regs, student_feats = self.model.forward_logits(images)
                
                feat_sizes = [(f.size(2), f.size(3)) for f in student_feats]
                priors = self.model.anchor_generator.generate(feat_sizes, self.model.img_size, student_logits.device)
                
                gt_loss_dict = self.model.multibox_loss(student_logits, student_regs, original_targets, priors)
                
                loss = gt_loss_dict["bbox_regression"] + gt_loss_dict["classification"]

                total_val_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())

                # Post-process model outputs for mAP calculation
                # Ensure original_targets passed to post_process have img_size in them if needed
                # self.model.post_process expects `original_targets` to potentially contain image metadata like original image size if transformations were applied
                detections = self.model.post_process(student_logits, student_regs, priors, self.model.img_size, self.config['score_thresh'])
                
                for i in range(len(detections)):
                    # Ensure boxes are float32 and labels/scores are int64/float32
                    # Detections might be empty if no boxes above threshold
                    pred_boxes = detections[i][:, :4] if detections[i].numel() > 0 else torch.empty((0, 4), device=self.device, dtype=torch.float32)
                    pred_scores = detections[i][:, 4] if detections[i].numel() > 0 else torch.empty((0,), device=self.device, dtype=torch.float32)
                    pred_labels = detections[i][:, 5].long() if detections[i].numel() > 0 else torch.empty((0,), device=self.device, dtype=torch.int64)

                    all_preds.append({
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    })

                    all_targets.append({
                        "boxes": original_targets[i]["boxes"],
                        "labels": original_targets[i]["labels"],
                    })
        
        # Compute mAP, Precision, Recall
        avg_val_loss = total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

        if len(all_preds) > 0 and len(all_targets) > 0:
            self.map_metric.update(all_preds, all_targets)
            map_results = self.map_metric.compute()
            
            mAP_0_5 = map_results['map_50'].item() if 'map_50' in map_results else 0.0
            # Note: torchmetrics v0.11 doesn't directly provide overall P and R for object detection.
            # map_50_recall might be related to average recall, or can be derived from other parts of map_results.
            # For simplicity, we'll use mAP_50 as the primary metric for early stopping and report it.
            # You would typically need to iterate through map_results['per_class_stats'] or use custom logic
            # to get overall precision/recall if class_metrics=True was used.
            
            # Placeholder for overall Precision and Recall - direct computation is complex
            # and might require custom aggregation from per-class metrics if needed precisely.
            # For this task, we focus on mAP_0_5 for early stopping and general evaluation.
            precision_0_5 = mAP_0_5 # Simplified for display
            recall_0_5 = mAP_0_5 # Simplified for display

            self.map_metric.reset() # Reset for next epoch
        else:
            mAP_0_5 = 0.0
            precision_0_5 = 0.0
            recall_0_5 = 0.0
            print("Warning: No valid predictions or targets for mAP calculation in this epoch. Setting metrics to 0.")

        # Return relevant metrics
        return avg_val_loss, mAP_0_5, precision_0_5, recall_0_5

    def save_checkpoint(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.feature_adapters:
            state['adapters_state_dict'] = self.feature_adapters.state_dict()

        torch.save(state, path)
        print(f"Saved checkpoint to {path}")
