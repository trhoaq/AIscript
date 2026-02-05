import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- 1. Cấu hình ---
DATA_ROOT = '../DatasetVoc/f'
BATCH_SIZE = 12
NUM_CLASSES = 1
LR = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách 20 lớp của Pascal VOC
VOC_CLASSES = ['f']
class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}

# --- 2. Custom Dataset để đọc cấu trúc VOC ---
class PascalVOCDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.transform = transform
        self.images_path = os.path.join(root, 'Image')
        self.anno_path = os.path.join(root, 'Annotations')
        
        # Đọc danh sách file từ ImageSets
        set_file = os.path.join(root, 'ImageSets/Main', f'{image_set}.txt')
        self.ids = []
        with open(set_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean_name = line.strip()
                clean_name = clean_name.replace('\ufeff', '').replace('\ufefe','')
                if clean_name:
                    self.ids.append(clean_name)

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # Sử dụng os.path.join để đảm bảo đường dẫn chuẩn trên cả Windows/Linux
        img_full_path = os.path.join(self.images_path, f'{img_id}.jpg')
        xml_full_path = os.path.join(self.anno_path, f'{img_id}.xml')

        # Kiểm tra file tồn tại để tránh crash giữa chừng
        if not os.path.exists(img_full_path):
            print(f"Warning: Image file not found {img_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))
        
        if not os.path.exists(xml_full_path):
            print(f"Warning: Annotation file not found {xml_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        img = Image.open(img_full_path).convert('RGB')
        
        # Parse XML
        tree = ET.parse(xml_full_path)
        root_xml = tree.getroot()
        
        # Lấy nhãn của object đầu tiên
        obj = root_xml.find('object')
        if obj is None:
            print(f"Warning: No 'object' found in {xml_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))
            
        label_name = obj.find('name').text.strip()
        label = class_to_idx[label_name]

        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.ids)

# --- 3. Preprocessing & Loaders ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = PascalVOCDataset(DATA_ROOT, 'default', transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- 4. Khởi tạo Model MobileNetV2 từ timm ---
# Sử dụng 'mobilenetv2_100' là bản chuẩn
model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=NUM_CLASSES)
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