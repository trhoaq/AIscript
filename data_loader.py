import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore
import cv2 # pyright: ignore[reportMissingImports]
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset # pyright: ignore[reportMissingImports]
import os, json

with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

OBJ_CLASSES = CFG["obj_classes"]
# Mặc định dùng 3 lớp trong VOC_CLASSES2 để phù hợp với NUM_CLASSES = 3
class_to_idx = {cls: i for i, cls in enumerate(OBJ_CLASSES)}
def get_train_preprocess(img_size=512):
    """
    Preprocess cho train: resize + padding, normalize.
    """
def get_train_transforms(img_size=512):
    """
    Augment cho train: resize + padding, lật, biến đổi màu, blur nhẹ, normalize.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=1, max_height=img_size // 8, max_width=img_size // 8, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

def get_eval_transforms(img_size=512):
    """
    Transform cho eval: resize + padding + normalize.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

class PascalVOCDataset(Dataset):
    """
    Dataset tùy chỉnh để tải dữ liệu từ tập dữ liệu Pascal VOC.

    Attributes:
        root (str): Đường dẫn gốc của tập dữ liệu Pascal VOC.
        mode (str): Chế độ mở ảnh (ví dụ: 'RGB').
        preprocessing (callable, optional): Hàm tiền xử lý áp dụng cho ảnh sau khi đọc.
        transform (callable, optional): Hàm biến đổi áp dụng cho ảnh và bounding box.
        images_path (str): Đường dẫn đến thư mục chứa ảnh.
        anno_path (str): Đường dẫn đến thư mục chứa các tệp chú thích XML.
        ids (list): Danh sách các ID ảnh được tải từ tệp image_set.
    """
    def __init__(self, root, image_set='default', mode='RGB', preprocessing=None, transform=None):
        """
        Khởi tạo PascalVOCDataset.

        Args:
            root (str): Đường dẫn gốc của tập dữ liệu Pascal VOC (chứa Image, Annotations, ImageSets).
            image_set (str): Tên của tập hợp ảnh (ví dụ: 'train', 'val', 'default').
                             Tệp văn bản chứa danh sách ID ảnh sẽ được đọc từ
                             `root/ImageSets/Main/{image_set}.txt`.
            mode (str): Chế độ mở ảnh, ví dụ: 'RGB' cho ảnh màu.
            preprocessing (callable, optional): Một hàm (ví dụ: từ Albumentations) sẽ được áp dụng
                                                cho ảnh sau khi đọc nhưng trước khi trả về.
            transform (callable, optional): Một hàm (ví dụ: từ Albumentations) sẽ được áp dụng
                                            cho ảnh và chú thích sau khi tiền xử lý.
        """
        self.root = root
        self.mode = mode
        self.preprocessing = preprocessing
        self.transform = transform
        self.images_path = os.path.join(root, 'Image')
        self.anno_path = os.path.join(root, 'Annotations')
        
        set_file = os.path.join(root, 'ImageSets/Main', f'{image_set}.txt')
        self.ids = []
        with open(set_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                clean_name = line.strip()
                clean_name = clean_name.replace('\ufeff', '').replace('\ufefe','')
                if clean_name:
                    self.ids.append(clean_name)

    def __getitem__(self, index):
        """
        Lấy một mục dữ liệu từ dataset tại chỉ mục đã cho.

        Args:
            index (int): Chỉ mục của mục dữ liệu cần lấy.

        Returns:
            tuple: Một tuple chứa ảnh đã tiền xử lý và nhãn tương ứng.
                   Nếu có lỗi (ví dụ: tệp không tồn tại, không có đối tượng trong XML),
                   nó sẽ in cảnh báo và trả về mục tiếp theo.
        """
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

        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Failed to read image {img_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))
        if self.mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Parse XML
        tree = ET.parse(xml_full_path)
        root_xml = tree.getroot()
        
        objects = root_xml.findall('object')
        if not objects:
            print(f"Warning: No 'object' found in {xml_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        boxes = []
        labels = []
        for obj in objects:
            name_tag = obj.find('name')
            if name_tag is None:
                continue
            label_name = name_tag.text.strip()
            if label_name in class_to_idx:
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_to_idx[label_name])

        if len(boxes) == 0:
            print(f"Warning: No valid class found in {xml_full_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        elif self.preprocessing is not None:
            transformed = self.preprocessing(image=img, bboxes=boxes, class_labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            transformed = get_eval_transforms()(image=img, bboxes=boxes, class_labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        return img, {
            "boxes": boxes,
            "labels": labels,
        }

    def __len__(self):
        """
        Trả về số lượng mẫu trong dataset.
        Returns:
            int: Tổng số ảnh trong dataset.
        """
        return len(self.ids)
