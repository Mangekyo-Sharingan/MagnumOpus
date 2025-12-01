# resnet50 - Trained Model Package

## Model Information
- **Model Architecture**: ResNet50Model
- **Training Date**: 20251121_145204
- **Number of Classes**: 5
- **Best Validation Loss**: 0.1153
- **Best Validation Accuracy**: 0.7900

## Training Summary
- **Total Epochs**: 50
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Total Training Time**: 7h 22m
- **Device**: cuda

## Files in this Package
1. `resnet50_model.pth` - Model weights and architecture info
2. `resnet50_training_history.npy` - Complete training history
3. `resnet50_optimizer.pth` - Optimizer state (for resume training)
4. `resnet50_metadata.json` - Training metadata and metrics
5. `resnet50_training_plots.png` - Training/validation curves
6. `README.md` - This file

## How to Load and Use for Predictions

```python
import torch
from modules import ModelFactory, Config

# Load configuration
config = Config()

# Create model instance
model = ModelFactory.create_model('resnet50', config)

# Load trained weights
checkpoint = torch.load('resnet50_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

## Performance Metrics
- Final Training Loss: 0.1165
- Final Training Accuracy: 0.7550
- Final Validation Loss: 0.1153
- Final Validation Accuracy: 0.7900
