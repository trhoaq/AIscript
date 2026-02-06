import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

from tqdm import tqdm

from data_loader import PascalVOCDataset

# --- 1. Cấu hình ---
DATA_ROOT = './Dataset/wep'
BATCH_SIZE = 16
NUM_CLASSES = 3
LR = 0.001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách 20 lớp của Pascal VOC
VOC_CLASSES1, VOC_CLASSES2 = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14'],['f','h','s']
class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES1)}

# --- 2. Custom Dataset để đọc cấu trúc VOC ---


# --- 3. Preprocessing & Loaders ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = PascalVOCDataset(DATA_ROOT, 'default', transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- 4. Khởi tạo Model MobileNetV2 từ timm ---
# Sử dụng 'mobilenetv2_100' là bản chuẩn
model = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# --- 5. Loss và Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 6. Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# --- 7. Save ---
torch.save(model.state_dict(), 'mobilenetv2_voc_final.pth')
print("Training Complete!")