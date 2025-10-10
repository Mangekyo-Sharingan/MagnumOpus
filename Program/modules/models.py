"""
CNN Models module for diabetic retinopathy classification
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all CNN models"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None

    @abstractmethod
    def build_model(self):
        """Build the CNN architecture"""
        pass

    @abstractmethod
    def compile_model(self):
        """Compile the model with optimizer, loss, and metrics"""
        pass

    def get_model_summary(self):
        """Get model architecture summary"""
        pass

    def save_model(self, filepath):
        """Save trained model"""
        pass

    def load_model(self, filepath):
        """Load pre-trained model"""
        pass

class VGG16Model(BaseModel):
    """VGG16 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "VGG16"

    def build_model(self):
        """Build VGG16 architecture"""
        pass

    def compile_model(self):
        """Compile VGG16 model"""
        pass

class ResNetModel(BaseModel):
    """ResNet50 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "ResNet50"

    def build_model(self):
        """Build ResNet50 architecture"""
        pass

    def compile_model(self):
        """Compile ResNet50 model"""
        pass

class InceptionV3Model(BaseModel):
    """InceptionV3 model implementation for diabetic retinopathy classification"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = "InceptionV3"

    def build_model(self):
        """Build InceptionV3 architecture"""
        pass

    def compile_model(self):
        """Compile InceptionV3 model"""
        pass

class ModelFactory:
    """Factory class for creating different CNN models"""

    @staticmethod
    def create_model(model_name, config):
        """Create and return the specified model"""
        model_map = {
            'vgg16': VGG16Model,
            'resnet50': ResNetModel,
            'inceptionv3': InceptionV3Model
        }

        if model_name.lower() not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        return model_map[model_name.lower()](config)

    @staticmethod
    def get_available_models():
        """Return list of available models"""
        return ['vgg16', 'resnet50', 'inceptionv3']
