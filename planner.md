# Ultra-Light Object Detection - TODO List

## 1. Model Architecture
- [x] **Backbone**: Implement GhostNet 0.5x using `timm`.
- [x] **Neck**: Build a lightweight FPN with Partial Convolution (PConv).
- [ ] Experiment with FPN channel sizes (48 or 64).
- [x] **Head**: Create an SSD Lite head using Depthwise Separable Convolutions.
- [ ] **Analysis**: (Optional) Evaluate removing anchors from the smallest feature map (stride 8).

## 2. Training Strategy
- [x] **Knowledge Distillation**:
  - [x] Select and implement a teacher model (SSDMobile).
  - [x] Implement distillation loss for both feature maps (FPN) and final logits.
- [x] **Data Augmentation**:
  - [x] Mosaic 4-mix (Implemented correctly in `data_loader.py`).
  - [x] Mixup (alpha ~0.2) (Implemented correctly in `data_loader.py`).
  - [x] HSV Shift (Implemented in `data_loader.py`).
  - [x] Random Affine (Shift, Scale, Rotate) (Implemented in `data_loader.py`).
  - [x] Add `Blur` augmentation to the pipeline (GaussianBlur exists).
- [x] **Training Loop**:
  - [x] Develop the main training script (`train.py`).
  - [x] Integrate the custom model and data loaders.
  - [x] Implement the distillation process within the training loop.

## 3. Deployment & Optimization
- [x] **Export**: Create a script to export the trained model to ONNX format (opset=18).
- [x] **Quantization**:
  - [x] Use OpenVINO NNCF to perform Post-Training Quantization (PTQ).
  - [x] Calibrate using the KL-Divergence algorithm.
- [x] **Inference**:
  - [x] Write an inference script using OpenVINO.
  - [x] Configure the runtime for latency (`PerformanceMode::LATENCY`).
  - [x] Set inference threads to 2.
