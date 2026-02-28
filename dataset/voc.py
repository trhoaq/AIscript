import os
import shutil
import tarfile
import requests
from tqdm import tqdm
from data_loader import PascalVOCDataset, get_train_transforms, get_eval_transforms # Assuming PascalVOCDataset is in data_loader.py

# Base URL for VOC datasets
VOC_URLS = {
    '2012_trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2012.tar',
}

def download_file(url, path):
    """Downloads a file from a URL to a given path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 KB
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=url.split('/')[-1])
    with open(path, 'wb') as f:
        for chunk in response.iter_content(block_size):
            t.update(len(chunk))
            f.write(chunk)
    t.close()

def extract_tar(file_path, dest_dir):
    """Extracts a tar file to a destination directory."""
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(dest_dir)

def prepare_voc_dataset(year='2012', root_dir='./data/VOC', download=True):
    """
    Prepares the Pascal VOC dataset (downloads and extracts if necessary).
    
    Args:
        year (str): The VOC year, e.g., '2012', '2007'.
        root_dir (str): The root directory where VOCdevkit will be stored.
        download (bool): Whether to download the dataset if not found.
    """
    voc_devkit_dir = os.path.join(root_dir, f'VOCdevkit/VOC{year}')
    
    if os.path.exists(voc_devkit_dir):
        print(f"VOC{year} dataset already exists at {voc_devkit_dir}. Skipping download.")
        return

    if not download:
        raise FileNotFoundError(f"VOC{year} dataset not found at {voc_devkit_dir}. Set download=True to download.")

    os.makedirs(root_dir, exist_ok=True)

    # Handle VOC2012 train/val
    if year == '2012':
        tar_url = VOC_URLS['2012_trainval']
        tar_path = os.path.join(root_dir, 'VOCdevkit_2012.tar')
        print(f"Downloading VOC2012 train/val from {tar_url}...")
        download_file(tar_url, tar_path)
        print(f"Extracting {tar_path}...")
        extract_tar(tar_path, root_dir)
        os.remove(tar_path)
        print("VOC2012 train/val prepared.")
    else:
        raise ValueError(f"Unsupported VOC year: {year}. Only '2012' is supported.")

def get_voc_datasets(voc_root='./data/VOC', img_size=256, years=['2012'], transform_train=None, transform_val=None):
    """
    Gets Pascal VOC datasets for specified years, combining train/val sets as needed.
    
    Args:
        voc_root (str): Root directory for VOC datasets.
        img_size (int): Image size for transformations.
        years (list): List of VOC years to use (e.g., ['2012', '2007']).
        transform_train: Training transformations. Defaults to get_train_transforms(img_size).
        transform_val: Validation transformations. Defaults to get_eval_transforms(img_size).
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if transform_train is None:
        transform_train = get_train_transforms(img_size)
    if transform_val is None:
        transform_val = get_eval_transforms(img_size)

    train_datasets = []
    val_datasets = []

    for year in years:
        print(f"Preparing VOC{year}...")
        prepare_voc_dataset(year, voc_root)
        
        # PascalVOCDataset expects the root to be VOCdevkit/VOC{year}
        dataset_base_path = os.path.join(voc_root, f'VOCdevkit/VOC{year}')
        
        # VOC2012 only has train/val split
        if year == '2012':
            train_datasets.append(PascalVOCDataset(root=dataset_base_path, image_set='train', transform=transform_train))
            val_datasets.append(PascalVOCDataset(root=dataset_base_path, image_set='val', transform=transform_val))
        else:
            raise ValueError(f"Unsupported VOC year: {year}. Only '2012' is supported in get_voc_datasets.")
    
    # Concatenate datasets if multiple years are specified
    final_train_dataset = torch.utils.data.ConcatDataset(train_datasets) if len(train_datasets) > 1 else (train_datasets[0] if train_datasets else None)
    final_val_dataset = torch.utils.data.ConcatDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)

    return final_train_dataset, final_val_dataset

if __name__ == '__main__':
    # Example usage:
    # This will download and prepare VOC2012 and VOC2007,
    # then create combined train/val datasets.
    train_ds, val_ds = get_voc_datasets(years=['2012', '2007'])
    
    if train_ds:
        print(f"Combined Train Dataset size: {len(train_ds)}")
    if val_ds:
        print(f"Combined Val Dataset size: {len(val_ds)}")

    # You can then use these datasets with DataLoader
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    # val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
    
    # for i, (images, targets) in enumerate(train_loader):
    #     print(f"Batch {i}: images shape {images.shape}, targets len {len(targets)}")
    #     break
