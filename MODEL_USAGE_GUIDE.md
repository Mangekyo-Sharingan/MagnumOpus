# Model Saving and Loading Guide

## Overview
Your training system now automatically saves complete model packages after each model finishes training. Each package contains everything needed to load the model later and make predictions.

## Training with Automatic Saving and Pause

When you run training, the system will:
1. Train each model completely
2. **Automatically save a complete model package** with all files needed for future use
3. **Pause and wait for your input** before continuing to the next model
4. Allow you to skip remaining models if needed

### Example Training Flow:
```
====================================================================================================
  TRAINING MODEL 1/3: VGG16
====================================================================================================

[Training happens with progress tracking...]

====================================================================================================
💾 SAVING COMPLETE MODEL PACKAGE: VGG16
====================================================================================================
✓ Model weights saved: VGG16_model.pth
✓ Training history saved: VGG16_training_history.npy
✓ Optimizer state saved: VGG16_optimizer.pth
✓ Metadata saved: VGG16_metadata.json
✓ Training plots saved: VGG16_training_plots.png
✓ README created: README.md

====================================================================================================
✅ VGG16 TRAINING AND SAVING COMPLETED!
====================================================================================================
📦 Model package location: C:\Users\adamh\PycharmProjects\MagnumOpus\models\VGG16_20241110_143022
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
⏸️  PAUSED: VGG16 training complete.
────────────────────────────────────────────────────────────────────────────────────────────────────
📊 Models completed: 1/3
📋 Remaining models: ResNet50, InceptionV3
────────────────────────────────────────────────────────────────────────────────────────────────────

▶️  Press ENTER to continue training ResNet50 or type 'skip' to end training: 
```

## What Gets Saved

Each model package contains:

### 1. **Model Weights** (`ModelName_model.pth`)
- Trained model state dictionary
- Model architecture information
- Number of classes
- Input size requirements

### 2. **Training History** (`ModelName_training_history.npy`)
- Loss and accuracy for each epoch
- Learning rates
- Training times
- Memory usage

### 3. **Optimizer State** (`ModelName_optimizer.pth`)
- Optimizer configuration (for resuming training if needed)
- Learning rate scheduler state

### 4. **Metadata** (`ModelName_metadata.json`)
```json
{
    "model_name": "VGG16",
    "model_class": "VGG16Model",
    "num_classes": 5,
    "training_date": "20241110_143022",
    "total_epochs_trained": 50,
    "best_val_loss": 0.2345,
    "best_val_acc": 0.8923,
    "batch_size": 32,
    "learning_rate": 0.001,
    "device": "cuda",
    "total_training_time": 16532.45
}
```

### 5. **Training Plots** (`ModelName_training_plots.png`)
- Visual representation of training/validation curves

### 6. **README.md**
- Instructions on how to load and use the model
- Performance metrics
- Training configuration details

## Loading Models for Predictions

### Method 1: Simple Loading
```python
from modules import load_model_for_prediction

# Load the model
model_loader = load_model_for_prediction('models/VGG16_20241110_143022')

# Make a prediction
result = model_loader.predict_with_details('path/to/test_image.png')
```

**Output:**
```
============================================================
PREDICTION RESULTS
============================================================
Image: test_image.png
Predicted Class: Moderate
Confidence: 87.34%

All Class Probabilities:
------------------------------------------------------------
  No DR                      12.45%
  Mild                        5.32%
→ Moderate                   87.34%
  Severe                      3.21%
  Proliferative DR            1.68%
============================================================
```

### Method 2: Batch Predictions
```python
from modules import load_model_for_prediction

# Load the model
model_loader = load_model_for_prediction('models/ResNet50_20241110_150000')

# Predict on multiple images
image_paths = [
    'Data/Aptos/test_images/0005cfc8afb6.png',
    'Data/Aptos/test_images/003f0afdcd15.png',
    'Data/Aptos/test_images/006efc72b638.png'
]

results = model_loader.predict_batch(image_paths)

# Process results
for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['class_names'][result['predicted_class']]}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("-" * 60)
```

### Method 3: Using ModelLoader Class Directly
```python
from modules import ModelLoader

# Initialize loader
loader = ModelLoader('models/InceptionV3_20241110_160000')

# Get model information
loader.get_model_info()

# Make prediction
result = loader.predict_image('test_image.png')

# Access prediction details
predicted_class = result['predicted_class']
confidence = result['confidence']
probabilities = result['probabilities']
class_names = result['class_names']

print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence * 100:.2f}%")
```

### Method 4: Compare Multiple Models
```python
from modules import compare_models_prediction

# Compare predictions from all trained models
model_dirs = [
    'models/VGG16_20241110_143022',
    'models/ResNet50_20241110_150000',
    'models/InceptionV3_20241110_160000'
]

results = compare_models_prediction(model_dirs, 'test_image.png')
```

**Output:**
```
================================================================================
COMPARING MULTIPLE MODELS
================================================================================
Image: test_image.png

Model           Prediction                Confidence   Val Accuracy   
--------------------------------------------------------------------------------
VGG16           Moderate                    87.34%        89.23%
ResNet50        Moderate                    92.15%        91.45%
InceptionV3     Severe                      78.92%        90.12%
================================================================================
```

## Integration in Your Own Scripts

### Standalone Prediction Script
```python
# predict.py
import sys
from pathlib import Path
from modules import load_model_for_prediction

def main():
    # Configuration
    model_path = 'models/VGG16_20241110_143022'  # Your trained model
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_image.png'
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model_for_prediction(model_path)
    
    # Show model info
    model.get_model_info()
    
    # Make prediction
    result = model.predict_with_details(image_path)
    
    # Additional processing
    if result['confidence'] > 0.8:
        print(f"✓ High confidence prediction!")
    else:
        print(f"⚠️ Low confidence - consider reviewing manually")

if __name__ == "__main__":
    main()
```

### Batch Processing Script
```python
# batch_predict.py
from pathlib import Path
from modules import load_model_for_prediction
import pandas as pd

def process_directory(model_path, image_dir, output_csv):
    """Process all images in a directory and save results"""
    
    # Load model
    model = load_model_for_prediction(model_path)
    
    # Get all images
    image_paths = list(Path(image_dir).glob('*.png'))
    
    # Make predictions
    print(f"Processing {len(image_paths)} images...")
    results = model.predict_batch(image_paths)
    
    # Convert to DataFrame
    data = []
    for result in results:
        data.append({
            'image_name': Path(result['image_path']).name,
            'prediction': result['class_names'][result['predicted_class']],
            'confidence': result['confidence'],
            'class_0_prob': result['probabilities'][0],
            'class_1_prob': result['probabilities'][1],
            'class_2_prob': result['probabilities'][2],
            'class_3_prob': result['probabilities'][3],
            'class_4_prob': result['probabilities'][4],
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to {output_csv}")
    
    return df

# Usage
if __name__ == "__main__":
    results_df = process_directory(
        model_path='models/ResNet50_20241110_150000',
        image_dir='Data/Aptos/test_images',
        output_csv='predictions.csv'
    )
    
    print(results_df.head())
```

## Finding Your Saved Models

All models are saved in timestamped directories:
```
models/
├── VGG16_20241110_143022/
│   ├── VGG16_model.pth
│   ├── VGG16_training_history.npy
│   ├── VGG16_optimizer.pth
│   ├── VGG16_metadata.json
│   ├── VGG16_training_plots.png
│   └── README.md
├── ResNet50_20241110_150000/
│   ├── ResNet50_model.pth
│   ├── ...
└── InceptionV3_20241110_160000/
    ├── InceptionV3_model.pth
    └── ...
```

## Class Labels Reference

The models predict 5 classes for diabetic retinopathy severity:
- **Class 0**: No DR (No Diabetic Retinopathy)
- **Class 1**: Mild
- **Class 2**: Moderate
- **Class 3**: Severe
- **Class 4**: Proliferative DR

## Advanced Usage

### Resume Training from Saved Model
```python
from modules import Trainer, ModelFactory, Config
import torch

# Load config
config = Config()

# Create fresh model instance
model = ModelFactory.create_model('vgg16', config)

# Load saved model package
package_dir = 'models/VGG16_20241110_143022'
checkpoint = torch.load(f'{package_dir}/VGG16_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create trainer and optionally load optimizer state
trainer = Trainer(model, config)
trainer.setup_training_components()

# Load optimizer if resuming training
optimizer_checkpoint = torch.load(f'{package_dir}/VGG16_optimizer.pth')
trainer.optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])

# Continue training
# trainer.train(train_loader, val_loader, num_epochs=10)
```

## Tips

1. **Model Selection**: After training, check the metadata.json files to compare validation accuracies and choose the best model

2. **Confidence Thresholds**: For critical applications, set a confidence threshold (e.g., 80%) and flag low-confidence predictions for manual review

3. **Ensemble Predictions**: Use `compare_models_prediction()` to get predictions from multiple models and use voting or averaging

4. **Storage**: Each model package is typically 100-500 MB depending on the architecture

5. **Version Control**: The timestamp in the directory name helps track different training runs

## Troubleshooting

**Q: Model file not found?**
- Check the exact path and timestamp in the models directory

**Q: Import errors?**
- Make sure you're running from the Program directory or have it in your Python path

**Q: CUDA out of memory during prediction?**
- The model automatically uses CPU if CUDA is unavailable
- For batch predictions, process smaller batches

**Q: Different image sizes?**
- All images are automatically resized to 224x224 during preprocessing

