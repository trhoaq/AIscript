# Ultra-Light Object Detection with Knowledge Distillation

## Project Overview

This project implements an ultra-light object detection system optimized for resource-constrained environments. It leverages a GhostNet backbone for efficiency and incorporates Knowledge Distillation (KD) to transfer knowledge from a larger MobileNetV3-based teacher model to the smaller student model. The training pipeline includes advanced data augmentation techniques like Mosaic and MixUp, and is designed to train and evaluate on the Pascal VOC 2012 dataset.

## Features

-   **Model Architecture:**
    -   **Student Model (`SSDGhost`):** GhostNet 0.5x backbone with a lightweight FPN (using Partial Convolution) and an SSD Lite head (using Depthwise Separable Convolutions).
    -   **Teacher Model (`SSDMobile`):** MobileNetV3-Large backbone with a custom FPN and SSD-style detection heads.
-   **Training Strategy:**
    -   **Knowledge Distillation:** Implements distillation loss for both feature maps (using adapter layers to match dimensions) and final logits. Allows training the student with a pre-trained teacher.
    -   **Advanced Data Augmentation:** Supports Mosaic 4-mix, MixUp, HSV Shift, Random Affine (Shift, Scale, Rotate), GaussianBlur, and CoarseDropout.
    -   **Training Loop:** Dedicated scripts for training the teacher independently and the student using knowledge distillation. Includes a validation loop.
-   **Data Handling:** Automated download and preparation of Pascal VOC 2012 dataset. Robust data loading with error handling for corrupted samples.

## Project Structure

```
.
├── core/
│   ├── checkpoint_utils.py      # Shared checkpoint load/extract/find helpers
│   ├── config_utils.py          # Merged config loader
│   ├── data_loader.py           # Augmentation pipelines and collate function
│   ├── trainer.py               # Generic DetectorTrainer for training/evaluation
│   ├── wandb_utils.py           # Weights & Biases wrappers
│   ├── openvino_preprocess.py   # Shared OpenVINO image preprocess
│   └── openvino_runtime_utils.py# Shared OpenVINO runtime helpers
├── dataset/
│   └── voc.py              # Handles Pascal VOC dataset download and preparation
├── model/
│   ├── ghostnet.py         # GhostNet backbone implementation
│   ├── mobilenetv3_torch.py# MobileNetV3 implementation details
│   ├── ssdlite_ghostnet100.py # SSDGhost (student) model definition
│   ├── ssdlite_mobilenet.py# SSDMobile (teacher) model definition
│   └── utils.py            # Shared utility functions (bbox ops, anchor generation)
├── train.py                # Main script for training the SSDGhost (student) model with KD
├── train_teacher.py        # Script for independently training the SSDMobile (teacher) model
├── export_onnx.py          # Export to ONNX
├── quantize_openvino.py    # PTQ quantization to OpenVINO INT8
├── inference.py            # OpenVINO webcam inference
├── config/                 # Configuration files (training + dataset-specific)
├── requirements.txt        # List of Python dependencies
└── README.md               # Project README file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    py -m venv venv
    .\venv\Scripts\activate # On Windows
    source venv/bin/activate # On Linux/macOS
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (`config/config.json` + dataset configs)

Before training, you can configure training parameters in `config/config.json`.
Dataset-specific parameters live in `config/coco.json` or `config/voc.json` and are selected via `dataset_format`.

```json
{
    "obj_classes": ["background", "person", "car", "bicycle"],
    "lr": 0.001,
    "epochs": 100,
    "p_mosaic": 0.5,
    "p_mixup": 0.2,
    "img_size": 256,
    "batch_size": 32,
    "num_workers": 4,
    "voc_years": ["2012"],
    "voc_root": "./data/VOC",
    "weight_decay": 0.0001,
    "kd_feature_weight": 1.0,
    "kd_logit_weight": 1.0
}
```
-   `obj_classes`: List of object classes including 'background'.
-   `lr`: Learning rate.
-   `epochs`: Number of training epochs.
-   `p_mosaic`, `p_mixup`: Probabilities for Mosaic and MixUp augmentations.
-   `img_size`: Input image size for the models.
-   `batch_size`: Training batch size.
-   `num_workers`: Number of data loader workers.
-   `voc_years`: List of Pascal VOC years to use (e.g., `["2012"]`).
-   `voc_root`: Root directory where VOC dataset will be downloaded/expected.
-   `weight_decay`: Weight decay for the optimizer.
-   `kd_feature_weight`, `kd_logit_weight`: Weights for feature and logit distillation losses.

## Usage

### 1. Prepare Dataset

The dataset will be automatically downloaded and prepared the first time you run either `train_teacher.py` or `train.py`. Ensure you have internet access. The files will be stored in the directory specified by the dataset config in `config/`.

### 2. Train Teacher Model (SSDMobile)

It is highly recommended to pre-train the teacher model first to achieve good performance before using it for distillation.

```bash
py train_teacher.py
```
This script will save checkpoints of the trained `SSDMobile` model in the `models/` directory.

### 3. Train Student Model (SSDGhost) with Knowledge Distillation

After training the teacher, you can train the student model. You will need to load a pre-trained teacher checkpoint.

**TODO:** Add functionality to load a teacher checkpoint into `train.py`. Currently, `train.py` instantiates the teacher model from scratch.

```bash
py train.py
```
This script will train the `SSDGhost` model, guided by the `SSDMobile` teacher model, and save its checkpoints.

### 4. Deploy with OpenVINO (Export -> PTQ -> Webcam Inference)

1. Export PyTorch checkpoint to ONNX (opset 18):

```bash
venv\Scripts\python.exe export_onnx.py --model student
```

2. Quantize ONNX to INT8 OpenVINO IR with NNCF PTQ (KL/histogram calibration):

```bash
venv\Scripts\python.exe quantize_openvino.py --model student
```

3. Run webcam inference (`cam id=0`) with OpenVINO runtime in latency mode and 2 inference threads:

```bash
venv\Scripts\python.exe inference.py --model student --cam-id 0
```

Use `--model teacher` to run the teacher variant.

## Future Work (from planner.md)

-   **Model Architecture:**
    -   Experiment with FPN channel sizes (48 or 64).
    -   Evaluate removing anchors from the smallest feature map (stride 8).
-   **Performance Tuning:**
    -   Benchmarking and profiling.
    -   Implement FPS cap.
    -   Test long-term stability.
