import os
import torch # pyright: ignore[reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingImports]
import torch.optim as optim # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
from tqdm import tqdm

from data_loader import PascalVOCDataset, VOC_CLASSES2, get_train_transforms, get_eval_transforms
from mobilenetv3_torch import mobilenet_v3_large, load_pretrained_from_timm

# --- 1. Cấu hình ---pis
DATA_ROOT = '../DatasetVoc/f'
BATCH_SIZE = 16
NUM_CLASSES = len(VOC_CLASSES2)
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "train_log.txt"

# --- 2. Custom Dataset để đọc cấu trúc VOC ---
VOC_CLASSES1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
VOC_CLASSES2 = ['f', 'h', 's']

# --- 3. Preprocessing & Loaders ---
train_ds = PascalVOCDataset(DATA_ROOT, 'default', transform=get_train_transforms(512))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

val_set_file = os.path.join(DATA_ROOT, 'ImageSets', 'Main', 'val.txt')
val_loader = None
if os.path.exists(val_set_file):
    val_ds = PascalVOCDataset(DATA_ROOT, 'val', transform=get_eval_transforms(512))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# --- 4. Khởi tạo Model MobileNetV3 (custom) + load pretrained từ timm ---
model = mobilenet_v3_large(num_classes=NUM_CLASSES)
model = load_pretrained_from_timm(model, "mobilenetv3_large_100")
model = model.to(DEVICE)

# --- 5. Loss và Optimizer ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

# --- 6. Training Loop ---
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", dynamic_ncols=True)
    for images, labels in train_pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        lr = optimizer.param_groups[0]["lr"]
        train_pbar.set_postfix(
            loss=f"{total_loss / max(1, len(train_pbar)):.4f}",
            acc=f"{(correct / max(1, total)):.4f}",
            lr=f"{lr:.3e}",
        )

    train_acc = correct / max(1, total)
    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")

    if val_loader is not None:
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", dynamic_ncols=True)
            for images, labels in val_pbar:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix(
                    loss=f"{val_loss / max(1, len(val_pbar)):.4f}",
                    acc=f"{(val_correct / max(1, val_total)):.4f}",
                )
        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mobilenetv3_voc_best.pth')
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch+1}/{EPOCHS}, "
                f"train_loss={total_loss/len(train_loader):.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, "
                f"lr={lr:.6f}\n"
            )
    else:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch+1}/{EPOCHS}, "
                f"train_loss={total_loss/len(train_loader):.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"lr={lr:.6f}\n"
            )

# --- 7. Save ---
torch.save(model.state_dict(), 'mobilenetv3_voc_final.pth')
print("Training Complete!")
