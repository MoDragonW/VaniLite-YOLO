# VaniLite-YOLO: A Lightweight YOLO Model for Ginseng Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A lightweight YOLO-based object detection model for ginseng and herbal medicine detection, featuring VanillaNet architecture and structured pruning techniques.

</div>

## 📋 Overview

VaniLite-YOLO is a lightweight object detection framework specifically designed for ginseng and herbal medicine detection. It builds upon both YOLOv11 and YOLOv12 architectures, integrating VanillaNet blocks to achieve better efficiency. The project implements a structured pruning pipeline to further compress the model while maintaining high detection accuracy.

### Key Features

- **Dual-Baseline Architecture**: Based on both YOLOv11n and YOLOv12n with VanillaNet blocks
- **Structured Pruning**: Three-stage pruning pipeline for efficient model compression
- **Multi-class Detection**: Supports 17 classes of ginseng and herbal medicines
- **High Efficiency**: Optimized for edge devices and real-time applications
- **Easy to Use**: Simple training and inference pipeline

## 🎯 Supported Classes

The model can detect the following 17 classes:

| Training Label | Paper Name |
|---------------|------------|
| White Ginseng | Ginseng Radix et Rhizoma(White) |
| radix glehniae | Glehniae Radix |
| Salvia miltiorrhiza | Salviae Miltiorrhizae Radix et Rhizoma |
| Codonopsis pilosula | Codonopsis Radix |
| Korean ginseng | Panacis Ginseng Radix et Rhizoma Rubra(Korea) |
| red ginseng root | Panacis Ginseng Radix et Rhizoma Rubra |
| Sophora flavescens | Sophorae Flavescentis Radix |
| Changium smyrnioides | Changii Radix |
| Nansha ginseng | Adenophorae Radix |
| ginseng | Panacis Ginseng Radix et Rhizoma |
| Panax notoginseng | Notoginseng Radix et Rhizoma |
| Adenophora Root | Glehniae Radix |
| dendrobe | Dendrobii Caulis |
| radix pseudostellariae | Pseudostellariae Radix |
| Panax quinquefolius L | Panacis Quinquefolii Radix |
| Scrophularia | Scrophulariae Radix |
| bamboo | Polygonati Rhizoma |
## 📦 Project Structure

```
VaniLite-YOLO/
├── tools/                          # Pruning tools
│   ├── sparse_train.py            # Stage 1: Sparse training
│   ├── prune_detect_depgraph.py   # Stage 2: Structured pruning
│   └── finetune_pruned_detect.py  # Stage 3: Fine-tuning
├── trained_models/                 # Pre-trained models
│   ├── VaniLite-YOLO.pt           # Main model
│   ├── yolov12n-vanillanet.pt     # Baseline model
│   └── ...                        # Other variants
├── ONNX-Model/                     # ONNX exported models
├── data/                           # Dataset configuration
│   └── renshen.yaml               # Dataset config file
├── assets/                         # Sample images
├── mytrain.py                     # Training script
├── myval.py                       # Validation script
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- Conda environment

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/MoDragonW/VaniLite-YOLO.git
cd VaniLite-YOLO
```

2. **Create conda environment**

```bash
conda create -n YOLOv12-shen python=3.9 -y
conda activate YOLOv12-shen
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Training

#### Basic Training

```bash
python mytrain.py
```

#### Three-Stage Pruning Pipeline

**Stage 1: Sparse Training**

```bash
python tools/sparse_train.py
```

This stage applies L1 regularization to encourage weight sparsity.

**Stage 2: Structured Pruning**

```bash
python tools/prune_detect_depgraph.py
```

This stage removes unimportant channels based on BN scale.

**Stage 3: Fine-tuning**

```bash
python tools/finetune_pruned_detect.py
```

This stage recovers accuracy after pruning.

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('trained_models/VaniLite-YOLO.pt')

# Run inference
results = model('path/to/image.jpg')

# Visualize results
results[0].show()
```

### Validation

```bash
python myval.py
```

## 📊 Model Performance

| Model | Params | FLOPs | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-------|---------|--------------|
| YOLOv12n | 2.6M | 6.3G | 0.983 | 0.829 |
| YOLOv11n | 2.6M | 6.3G | 0.983 | 0.825 |
| VaniLite-YOLO | 0.8M | 2.4G | 0.977 | 0.790 |

*Performance on ginseng dataset*

> **Note**: VaniLite-YOLO is the final pruned and compressed model.

## 🔧 Configuration

### Dataset Preparation

1. Organize your dataset in YOLO format:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. Update `data/renshen.yaml` with your dataset paths:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

names:
  0: White Ginseng
  # ... add your classes
```

### Training Parameters

Modify training parameters in `mytrain.py`:

```python
model.train(
    data='data/renshen.yaml',
    epochs=800,
    imgsz=640,
    batch=-1,  # Auto batch size
    project='results',
    name='experiment_name'
)
```

## 📚 Paper Information

The technical details of VaniLite-YOLO are described in a research paper currently under review. This repository provides the official implementation of the model and pruning pipeline described in the paper.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the excellent YOLO implementation
- [VanillaNet](https://github.com/huawei-noah/VanillaNet) for the innovative architecture
- [torch-pruning](https://github.com/VainF/Torch-Pruning) for the structured pruning tools

## 📧 Contact

For questions and suggestions, please open an issue in the GitHub repository.

## 🔗 Links

- [Pre-trained Models](trained_models/) - Pre-trained model weights
- [ONNX Models](ONNX-Model/) - ONNX exported models for deployment

---

<div align="center">

**Made with ❤️ for Ginseng Detection**

</div>
