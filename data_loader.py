import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os

VOC_CLASSES1, VOC_CLASSES2 = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14'],['f','h','s']
class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES1)}

def pre_process():
    """
    Tạo pipeline tiền xử lý hình ảnh sử dụng thư viện Albumentations.
    Pipeline bao gồm các bước:
    - PadIfNeeded: Đệm ảnh nếu kích thước nhỏ hơn 4000x4000.
    - Resize: Thay đổi kích thước ảnh về 512x512.
    - Normalize: Chuẩn hóa giá trị pixel của ảnh.
    - ToTensorV2: Chuyển đổi ảnh sang định dạng Tensor của PyTorch.
    """
    pipeline=[
        A.PadIfNeeded(
            min_height=4000,
            min_width=4000,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
        A.Resize(
            height=512, 
            width=512, 
            p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]
    return A.Compose(pipeline)

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
    def __init__(self, root, image_set='default',mode='RGB', preprocessing=None, transform=None):
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

        img = self.preprocess(Image.open(img_full_path).convert(self.mode))
        
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
            
        return img, label

    def __len__(self):
        """
        Trả về số lượng mẫu trong dataset.

        Returns:
            int: Tổng số ảnh trong dataset.
        """
        return len(self.ids)


