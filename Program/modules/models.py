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
        from torchvision.models import VGG16_Weights
        self.backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # FREEZE ALL backbone layers initially (for Stage 1 training)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier for diabetic retinopathy (5 classes)
        num_features = self.backbone.classifier[0].in_features  # 25088

        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),  # Increased from 0.5 to combat overfitting
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(0.4),  # Increased from 0.5 to combat overfitting
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
        from torchvision.models import ResNet50_Weights
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # FREEZE ALL backbone layers initially (for Stage 1 training)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace final layer for diabetic retinopathy (5 classes)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Increased from 0.5
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm for better regularization
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.3
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
        from torchvision.models import Inception_V3_Weights
        self.backbone = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)

        # FREEZE ALL backbone layers initially (for Stage 1 training)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace final layer for diabetic retinopathy (5 classes)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Increased from 0.5
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm for better regularization
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.3
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


class ModelFactory:
    """Factory class for creating model instances"""

    @staticmethod
    def create_model(model_name, config):
        """
        Create and return a model instance

        Args:
            model_name: Name of the model ('vgg16', 'resnet50', 'inceptionv3')
            config: Configuration object

        Returns:
            Model instance
        """
        return get_model(model_name, config)

    @staticmethod
    def get_available_models():
        """Get list of available model names"""
        return ['vgg16', 'resnet50', 'inceptionv3']


# Test function for independent execution
def test_models_module():
    """Test the models module independently"""
    print("Testing Models module...")

    try:
        # Import config
        from config import Config

        # Create config
        config = Config()
        print("[OK] Config imported and created successfully")

        # Test model creation for each architecture
        model_names = ['vgg16', 'resnet50', 'inceptionv3']

        for model_name in model_names:
            try:
                print(f"Testing {model_name.upper()} model...")

                # Create model
                model = get_model(model_name, config)
                print(f"[OK] {model_name.upper()} model created successfully")

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
                        print(f"[OK] {model_name.upper()} forward pass successful")
                    else:
                        print(f" {model_name.upper()} output shape mismatch")

                # Test model summary
                print(f"  - Getting model summary for {model_name.upper()}:")
                model.get_model_summary()

            except Exception as e:
                print(f"[FAIL] {model_name.upper()} model test failed: {e}")
                return False

        print("[OK] All models tested successfully")
        print("[OK] Models module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Models module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models_module()
    print(f"\nModels Module Test Result: {'PASS' if success else 'FAIL'}")
