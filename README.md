
# Structural Damage Analyzer - Complete Training & Prediction Suite

A comprehensive GUI application for training and analyzing structural damage in images using U-Net segmentation models with both PyTorch and Keras implementations.

## Features

### 🎨 Modern Dark UI
- **Professional Interface**: Sleek dark mode design with custom logo
- **Colorful Buttons**: Color-coded buttons for different actions (success, warning, info, etc.)
- **Enhanced Typography**: Modern fonts and improved readability
- **Responsive Layout**: Adaptive interface that works on different screen sizes
- **Visual Feedback**: Hover effects and status indicators

### 🚀 Model Training

- **Dual Framework Support**: Train models using PyTorch (segmentation-models-pytorch) or Keras (keras-unet-collection)
- **Multiple Backbones**: Choose from ResNet, VGG, DenseNet, MobileNet, and EfficientNet architectures
- **Real-time Training Monitoring**: Live progress tracking and training logs
- **Configurable Parameters**: Adjust epochs, batch size, and data directories
- **Automatic Model Saving**: Best models saved automatically during training

### 🔍 Damage Prediction

- **U-Net Segmentation**: Advanced semantic segmentation for precise damage detection
- **Multi-Model Support**: Load and compare PyTorch and Keras models simultaneously
- **Batch Processing**: Analyze entire directories of images automatically
- **Visual Results**: Generate overlay images showing detected damage areas
- **Damage Classification**: Categorizes damage as Manageable, Partially damaged, or Completely damaged

### 📊 OpenCV Analysis

- **Traditional Computer Vision**: White pixel detection method for immediate analysis
- **No Training Required**: Analyze images without pre-trained models
- **Baseline Comparison**: Compare ML results with traditional CV methods

## Installation

1. **Clone or download this repository**

2. **Install Python 3.8 or higher**

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Run the application:**

```bash
python app.py
```

2. **Choose your workflow:**
   - **Training Tab**: Train new models from your data
   - **Prediction Tab**: Use trained models for damage detection
   - **OpenCV Tab**: Quick analysis using traditional computer vision

## Training Your Own Models

### Data Preparation

1. **Images**: Place your raw images in a directory (e.g., `RAW_Images/`)
2. **Masks**: Place corresponding binary masks in another directory (e.g., `Masks/`)
   - Masks should be binary images where white pixels indicate damage
   - Use the OpenCV tab to generate masks from images with visible damage

### Training Process

1. Go to the **Training** tab
2. Select framework (PyTorch or Keras) and backbone architecture
3. Set training parameters (epochs, batch size)
4. Choose your data directories
5. Click **Start Training**
6. Monitor progress in real-time

### Supported Backbones

**PyTorch Options:**

- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- VGG: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- DenseNet: `densenet121`, `densenet169`, `densenet201`
- MobileNet: `mobilenet_v2`
- EfficientNet: `efficientnet-b0` through `efficientnet-b7`

**Keras Options:**

- ResNet: `ResNet50`, `ResNet101`, `ResNet152`, `ResNet50V2`, `ResNet101V2`, `ResNet152V2`
- VGG: `vgg16`, `vgg19`
- DenseNet: `densenet121`, `densenet169`, `densenet201`
- MobileNet: `MobileNetV2`

## Using Trained Models

### Single Image Analysis

1. Go to the **Prediction** tab
2. Load your trained model(s) - specify the correct backbone used during training
3. Click **Load Single Image** to select an image
4. Click **Analyze Current Image** to get results

### Batch Processing

1. Load your trained model(s)
2. Click **Batch Process Directory**
3. Select a folder containing images
4. Results will be saved automatically with masks and overlay images

## Damage Classification

The system uses area-based thresholds to classify damage:

- **No Damage**: 0 pixels
- **Manageable**: 1-5,025 pixels
- **Partially Damaged**: 5,026-17,670 pixels
- **Completely Damaged**: >17,671 pixels

_These thresholds are based on the original research and can be modified in the code._

## File Structure

```
├── app.py                    # Main GUI application
├── utils/                    # Core utilities
│   ├── __init__.py
│   ├── model_loader.py       # Model loading and prediction
│   ├── image_utils.py        # Image processing utilities
│   └── training_utils.py     # Training management
├── models/                   # Place your trained models here
├── RAW_Images/              # Training images (create this folder)
├── Masks/                   # Training masks (create this folder)
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Model Architecture Details

### PyTorch Implementation

- Uses `segmentation-models-pytorch` library
- U-Net architecture with ImageNet pretrained encoders
- Binary segmentation with sigmoid activation
- BCEWithLogitsLoss for training

### Keras Implementation

- Uses `keras-unet-collection` library
- U-Net++ architecture with ImageNet pretrained backbones
- Binary segmentation with sigmoid activation
- Binary crossentropy loss for training

## Integration with Your Existing Workflow

This GUI integrates the complete functionality from your GitHub repository:

- **Training scripts**: `train_pytorch_unet.py` and `train_keras_unet.py`
- **Prediction scripts**: `predict_pytorch.py` and `predict_keras.py`
- **OpenCV analysis**: `damage_analyzer.py`

All functionality is now available through an intuitive graphical interface with additional features like real-time monitoring and batch processing.

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Models will automatically fall back to CPU if CUDA is unavailable
2. **Memory Errors**: Reduce batch size if you encounter out-of-memory errors
3. **Model Loading Fails**: Ensure the backbone specified matches the one used during training
4. **Training Stops**: Check that your data directories contain matching image and mask files

### Performance Tips

- **GPU Training**: Use CUDA-compatible GPU for faster training
- **Batch Size**: Start with batch size 4, reduce if memory issues occur
- **Image Size**: Models use 512x512 input size for optimal performance
- **Data Quality**: Ensure masks accurately represent damage areas

### Getting Help

If you encounter issues:

1. Check the training log for detailed error messages
2. Verify your data directory structure and file formats
3. Ensure all dependencies are correctly installed
4. Try with a smaller dataset first to validate the setup

## Credits

Based on the comprehensive damage detection system from [Project_19](https://github.com/mlbkumar9/Project_19) with enhanced GUI interface and additional features for improved usability.

