"""
CNN Models module for diabetic retinopathy classification
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(ABC, nn.Module):
    """Abstract base class for all CNN models"""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def build_model(self):
        """Build the CNN architecture"""
        pass

    def get_model_summary(self):
        """Get model architecture summary"""
        print(f"Model: {self.__class__.__name__}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.__class__.__name__
        }, filepath)

    def load_model(self, filepath):
        """Load pre-trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

class VGG16Model(BaseModel):
    """VGG16 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "VGG16"
        self.build_model()

    def build_model(self):
        """Build VGG16 architecture with pre-trained weights"""
        # Load pre-trained VGG16
        self.backbone = models.vgg16(pretrained=True)

        # Freeze early layers (optional - can be made configurable)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Replace classifier
        num_features = self.backbone.classifier[6].in_features

        # Custom classifier for diabetic retinopathy
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, self.num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class ResNet50Model(BaseModel):
    """ResNet50 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "ResNet50"
        self.build_model()

    def build_model(self):
        """Build ResNet50 architecture with pre-trained weights"""
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)

        # Freeze early layers (optional)
        for param in list(self.backbone.parameters())[:-2]:
            param.requires_grad = False

        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class InceptionV3Model(BaseModel):
    """InceptionV3 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "InceptionV3"
        self.build_model()

    def build_model(self):
        """Build InceptionV3 architecture with pre-trained weights"""
        # Load pre-trained InceptionV3
        self.backbone = models.inception_v3(pretrained=True, aux_logits=True)

        # Freeze early layers (optional)
        for param in list(self.backbone.parameters())[:-4]:
            param.requires_grad = False

        # Replace final layers
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

        # Replace auxiliary classifier if present
        if self.backbone.aux_logits:
            num_aux_features = self.backbone.AuxLogits.fc.in_features
            self.backbone.AuxLogits.fc = nn.Linear(num_aux_features, self.num_classes)

    def forward(self, x):
        if self.training and self.backbone.aux_logits:
            outputs, aux_outputs = self.backbone(x)
            return outputs, aux_outputs
        else:
            return self.backbone(x)

def get_model(model_name, config):
    """Factory function to get model instance"""
    models_dict = {
        'vgg16': VGG16Model,
        'resnet50': ResNet50Model,
        'inceptionv3': InceptionV3Model
    }

    if model_name.lower() not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models_dict.keys())}")

    return models_dict[model_name.lower()](config)

# Test function for independent execution
def test_models_module():
    """Test the models module independently"""
    print("Testing Models module...")

    try:
        # Import config
        from config import Config

        # Create config
        config = Config()
        print("✓ Config imported and created successfully")

        # Test model creation for each architecture
        model_names = ['vgg16', 'resnet50', 'inceptionv3']

        for model_name in model_names:
            try:
                print(f"Testing {model_name.upper()} model...")

                # Create model
                model = get_model(model_name, config)
                print(f"✓ {model_name.upper()} model created successfully")

                # Test model properties
                print(f"  - Model name: {model.model_name}")
                print(f"  - Number of classes: {model.num_classes}")
                print(f"  - Device: {model.device}")

                # Test forward pass with dummy data
                model.eval()
                if model_name == 'inceptionv3':
                    dummy_input = torch.randn(1, 3, 299, 299)
                else:
                    dummy_input = torch.randn(1, 3, 224, 224)

                with torch.no_grad():
                    output = model(dummy_input)
                    if isinstance(output, tuple):  # InceptionV3 with aux outputs
                        output = output[0]

                    print(f"  - Output shape: {output.shape}")
                    print(f"  - Expected shape: (1, {config.num_classes})")

                    if output.shape == (1, config.num_classes):
                        print(f"✓ {model_name.upper()} forward pass successful")
                    else:
                        print(f"⚠ {model_name.upper()} output shape mismatch")

                # Test model summary
                print(f"  - Getting model summary for {model_name.upper()}:")
                model.get_model_summary()

            except Exception as e:
                print(f"✗ {model_name.upper()} model test failed: {e}")
                return False

        print("✓ All models tested successfully")
        print("✓ Models module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Models module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models_module()
    print(f"\nModels Module Test Result: {'PASS' if success else 'FAIL'}")
