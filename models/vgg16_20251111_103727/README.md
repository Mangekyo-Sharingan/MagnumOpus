# vgg16 - Trained Model Package

## Model Information
- **Model Architecture**: VGG16Model
- **Training Date**: 20251111_103727
- **Number of Classes**: 5
- **Best Validation Loss**: 0.7794
- **Best Validation Accuracy**: 0.7365

## Training Summary
- **Total Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.005
- **Total Training Time**: 10h 5m
- **Device**: cuda

## Files in this Package
1. `vgg16_model.pth` - Model weights and architecture info
2. `vgg16_training_history.npy` - Complete training history
3. `vgg16_optimizer.pth` - Optimizer state (for resume training)
4. `vgg16_metadata.json` - Training metadata and metrics
5. `vgg16_training_plots.png` - Training/validation curves
6. `README.md` - This file

## How to Load and Use for Predictions

```python
import torch
from modules import ModelFactory, Config

# Load configuration
config = Config()

# Create model instance
model = ModelFactory.create_model('vgg16', config)

# Load trained weights
checkpoint = torch.load('vgg16_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

## Performance Metrics
- Final Training Loss: 0.8026
- Final Training Accuracy: 0.7368
- Final Validation Loss: 0.9488
- Final Validation Accuracy: 0.7330
