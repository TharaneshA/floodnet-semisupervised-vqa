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
- Image resizing to 224x224 pixels using Lanczos resampling
- Dataset organization into training, validation, and test sets
- Creation of metadata with dataset statistics
- Generation of sample images for verification

### 2. Flood Classification (classification.ipynb)

Implements a binary classification model to detect the presence of flooding in images:
- Semi-supervised learning approach leveraging unlabeled data
- Transfer learning with pre-trained models
- Performance evaluation on validation and test sets

### 3. Visual Question Answering (VQA)

Extends the model to answer specific questions about flood imagery:
- Integration with natural language processing
- Multi-modal learning combining image features and text queries
- Capability to answer questions about flood extent, damage assessment, etc.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Required packages: TensorFlow, PIL, NumPy, OpenCV, tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/floodnet-semisupervised-vqa.git
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
