import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from typing import List

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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader # Storing val_loader
        self.device = device
        self.config = config
        
        # Teacher model for Distillation
        self.teacher_model = config.get('teacher_model')
        self.distill_criterion = None
        self.feature_adapters = None
        
        params_to_optimize = list(model.parameters())

        if self.teacher_model:
            self.teacher_model.to(device)
            self.teacher_model.eval()
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
            return None

        self.model.eval()
        # Adapters should be in eval mode too if they exist
        if self.feature_adapters:
            self.feature_adapters.eval()

        total_val_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for batch in pbar:
                if batch[0] is None:
                    continue
                images, targets = batch
                
                images = torch.stack([img.to(self.device) for img in images])
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                student_logits, student_regs, student_feats = self.model.forward_logits(images)
                
                feat_sizes = [(f.size(2), f.size(3)) for f in student_feats]
                priors = self.model.anchor_generator.generate(feat_sizes, self.model.img_size, student_logits.device)
                
                gt_loss_dict = self.model.multibox_loss(student_logits, student_regs, targets, priors)
                
                loss = gt_loss_dict["bbox_regression"] + gt_loss_dict["classification"]

                # Note: No KD loss during evaluation typically
                total_val_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())
        
        return total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

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
