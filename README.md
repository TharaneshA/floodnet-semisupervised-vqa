# floodnet-semisupervised-vqa

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: FloodNet](https://img.shields.io/badge/Dataset-FloodNet-green.svg)](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)

## 🌊 Project Overview

This project addresses the FloodNet Challenge, focusing on flood detection and analysis using semi-supervised learning approaches. The project combines image classification for flood detection with Visual Question Answering (VQA) capabilities to extract detailed information from flood imagery.

## 📊 Dataset

The project utilizes the FloodNet dataset, which contains:

- **Labeled flood images**: Images with flood presence
- **Labeled non-flood images**: Images without flood presence
- **Unlabeled images**: Additional images without labels for semi-supervised learning
- **Validation and test sets**: For model evaluation

The dataset is organized into the following structure:
```
FloodNet_Challenge_Track1/
├── Train/
│   ├── Labeled/
│   │   ├── Flooded/
│   │   └── Non-Flooded/
│   └── Unlabeled/
├── Validation/
└── Test/
```

## 🔍 Project Components

### 1. Data Preprocessing (prep.ipynb)

The preprocessing pipeline includes:
- Image cropping from 4000px to 3000px width
- Image resizing to 512x512 pixels
- Dataset organization into training, validation, and test sets
- Creation of metadata with dataset statistics
- Generation of sample images for verification

### 2. Semantic Segmentation and Classification

Implements a dual-model approach for flood analysis:
- DeepLabV3+ model for semantic segmentation of flood regions
- EfficientNet for binary flood classification
- Semi-supervised learning leveraging unlabeled data
- Performance evaluation metrics and visualization

### 3. Visual Question Answering (VQA)

Custom VQA architecture for flood image analysis:
- ResNet18 backbone for image feature extraction
- LSTM network for question encoding
- Multi-modal fusion of image and text features
- Capability to answer questions about flood extent, damage assessment, etc.

### 4. Results and Visualization

Comprehensive output analysis:
- Segmentation masks highlighting flood regions
- Classification confidence scores
- Sample VQA responses with attention maps
- Performance metrics across all components

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Required packages: TensorFlow, PIL, NumPy, OpenCV, tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/TharaneshA/floodnet-semisupervised-vqa.git
cd floodnet-semisupervised-vqa

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the FloodNet Challenge dataset
2. Update the `DATA_ROOT` variable in `prep.ipynb` to point to your dataset location
3. Run the preprocessing notebook to prepare the data

## 📈 Model Training

1. Run the `classification.ipynb` notebook to train the flood detection model
2. The model will be trained using both labeled and unlabeled data in a semi-supervised approach

## 🔧 Future Work

- Implement advanced semi-supervised learning techniques
- Enhance VQA capabilities with more complex question types
- Explore multi-task learning approaches
- Deploy model for real-time flood monitoring

## 📚 References

- [FloodNet Challenge - EARTHVISION 2021](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)
- [Semi-Supervised Learning for Computer Vision](https://arxiv.org/abs/2006.10958)
- [Visual Question Answering: A Survey of Methods and Datasets](https://arxiv.org/abs/1607.05910)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
