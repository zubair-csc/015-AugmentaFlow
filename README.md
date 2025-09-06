# 🎨 Data Augmentation Pipeline with Albumentations

## 📋 Project Overview
This project implements a comprehensive and flexible data augmentation pipeline using Albumentations library in Python. Built with multiple augmentation strategies, it includes geometric transformations, color adjustments, noise/blur effects, weather simulations, and specialized pipelines for different computer vision tasks. The system supports both single image and batch processing, mask augmentation for segmentation tasks, and provides extensive visualization and configuration management capabilities.

## 🎯 Objectives
- Create robust data augmentation pipelines for computer vision tasks
- Implement multiple augmentation strategies (light, medium, heavy, segmentation)
- Support both classification and segmentation workflows with mask handling
- Provide configurable parameters for different augmentation techniques
- Enable batch processing and visualization of augmentation results
- Offer save/load functionality for augmentation configurations

## 📊 Dataset Information
| Attribute | Details |
|-----------|---------|
| **Input Format** | NumPy arrays (images and optional masks) |
| **Supported Types** | RGB/BGR images, Grayscale images, Segmentation masks |
| **Output Size** | Configurable (default: 256x256) |
| **Processing** | Single image, batch processing, or image-mask pairs |
| **Normalization** | ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

## 🔧 Technical Implementation

### 📌 Augmentation Techniques
- **Geometric Transformations**: Rotation, scaling, shifting, perspective changes, grid/optical distortions
- **Color Adjustments**: Brightness, contrast, saturation, hue modifications, CLAHE, gamma correction
- **Spatial Operations**: Horizontal/vertical flips, transpose, random crops, coarse dropout
- **Blur & Noise**: Gaussian blur, motion blur, noise injection, JPEG compression
- **Weather Effects**: Rain, fog, sun flares, random shadows
- **Advanced Features**: Channel shuffle, normalization, mask-aware transformations

### 🧹 Data Preprocessing
**Pipeline Processing:**
- Automatic image resizing to specified dimensions
- Probability-based augmentation selection using OneOf compositions
- Border handling for geometric transformations
- Mask synchronization for segmentation tasks
- Normalization with ImageNet statistics
- Support for various input formats (uint8, float32)

### ⚙️ Pipeline Architecture
**Four Specialized Pipelines:**
1. **Light Pipeline**: Minimal augmentations for validation/testing
   - Basic resizing and normalization
   - Light color adjustments
   - Horizontal flipping

2. **Medium Pipeline**: Balanced augmentations for general training
   - Moderate geometric transformations
   - Color jittering and HSV adjustments
   - Controlled probability settings

3. **Heavy Pipeline**: Aggressive augmentations for robust training
   - Advanced geometric distortions
   - Comprehensive color transformations
   - Noise, blur, and weather effects
   - Spatial cutouts and crops

4. **Segmentation Pipeline**: Specialized for segmentation tasks
   - Mask-aware transformations
   - Synchronized image-mask augmentation
   - Conservative parameters to preserve annotations

### 📏 Evaluation Features
**Configuration Management:**
- JSON-based configuration system
- Default and custom parameter support
- Save/load functionality for reproducibility
- Pipeline information and statistics

**Visualization Tools:**
- Side-by-side augmentation comparison
- Batch processing visualization
- Training sample generation
- Real-time augmentation preview

## 📊 Visualizations
- **Augmentation Comparison**: Original vs augmented images in grid layout
- **Pipeline Statistics**: Transform counts and probability distributions
- **Batch Processing**: Multiple augmented samples visualization
- **Configuration Display**: JSON-formatted parameter visualization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: Albumentations, OpenCV, NumPy, Matplotlib

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/015-AugmentaFlow.git
cd 015-AugmentaFlow
```

2. Install required libraries:
```bash
pip install albumentations opencv-python matplotlib numpy
```

### Dataset Setup
No external dataset required; the pipeline works with any image data loaded as NumPy arrays.

### Running the Pipeline
Execute the Python script:
```python
python augmentation_pipeline.py
```

This will:
- Initialize the augmentation pipeline
- Demonstrate single image and batch processing
- Show segmentation with mask support
- Display pipeline information and statistics
- Save configuration to JSON file

## 📈 Usage Examples

### Basic Usage
```python
# Initialize pipeline
augmenter = CustomAugmentationPipeline()

# Single image augmentation
aug_image = augmenter.augment(image, pipeline='medium')

# Batch processing
aug_batch = augmenter.augment_batch(image_list, pipeline='heavy')
```

### Segmentation with Masks
```python
# Augment image and mask together
result = augmenter.augment(image, pipeline='segmentation', mask=mask)
aug_image = result['image']
aug_mask = result['mask']
```

### Custom Configuration
```python
# Create custom configuration
custom_config = {
    'image_size': (224, 224),
    'geometric': {'rotate_limit': 45, 'probability': 0.8},
    # ... other parameters
}

# Initialize with custom config
custom_augmenter = CustomAugmentationPipeline(custom_config)
```

### Visualization
```python
# Visualize augmentation results
augmenter.visualize_augmentations(image, num_samples=4, pipeline='heavy')
```

## 🔮 Future Enhancements
- Integration with PyTorch and TensorFlow data loaders
- Advanced augmentation techniques (MixUp, CutMix, AutoAugment)
- Multi-scale and multi-resolution augmentation strategies
- Real-time augmentation parameter optimization
- Integration with popular computer vision frameworks
- Advanced weather and lighting effect simulations
- GPU-accelerated augmentation processing

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
- **Albumentations** for the powerful augmentation library
- **OpenCV** for computer vision operations
- **NumPy** and **Matplotlib** for numerical computing and visualization
- Open source community for continuous support and inspiration

## 📞 Contact
Zubair - [GitHub Profile](https://github.com/zubair-csc)

Project Link: [https://github.com/your-username/015-AugmentaFlow](https://github.com/your-username/015-AugmentaFlow)

⭐ Star this repository if you found it helpful!
