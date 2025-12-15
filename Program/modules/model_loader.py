"""
Model Loader module for loading trained models and making predictions
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ModelLoader:
    """Class for loading trained models and making predictions"""

    def __init__(self, model_package_dir):
        """
        Initialize the model loader

        Args:
            model_package_dir: Path to the saved model package directory
        """
        self.model_package_dir = Path(model_package_dir)
        self.model = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None

        # Load model automatically
        self._load_model()

    def _load_model(self):
        """Load the trained model from the package directory"""
        # Find metadata file
        metadata_files = list(self.model_package_dir.glob("*_metadata.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {self.model_package_dir}")

        metadata_path = metadata_files[0]
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loading model: {self.metadata['model_name']}")
        print(f"Model class: {self.metadata['model_class']}")
        print(f"Training date: {self.metadata['training_date']}")
        print(f"Best validation accuracy: {self.metadata['best_val_acc']:.4f}")

        # Find model weights file
        model_files = list(self.model_package_dir.glob("*_model.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {self.model_package_dir}")

        model_path = model_files[0]

        # Load model architecture
        try:
            from .config import Config
            from .models import ModelFactory
        except ImportError:
            from modules.config import Config
            from modules.models import ModelFactory
            
        config = Config()

        # Create model instance
        model_name = self.metadata['model_name'].lower()
        self.model = ModelFactory.create_model(model_name, config)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"[OK] Model loaded successfully on {self.device}")

        # Setup image transformations
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image preprocessing transformations"""
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        """
        Make prediction on a single image

        Args:
            image_path: Path to the image file

        Returns:
            dict: Prediction results including class, confidence, and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)

            # Handle InceptionV3 auxiliary outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Prepare results
        results = {
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'class_names': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        }

        return results

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images

        Args:
            image_paths: List of paths to image files

        Returns:
            list: List of prediction results for each image
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            result['image_path'] = str(image_path)
            results.append(result)

        return results

    def predict_with_details(self, image_path, show_all_probabilities=True):
        """
        Make detailed prediction with formatted output

        Args:
            image_path: Path to the image file
            show_all_probabilities: Whether to show probabilities for all classes

        Returns:
            dict: Detailed prediction results
        """
        result = self.predict_image(image_path)

        print(f"\n{'='*60}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"Image: {Path(image_path).name}")
        print(f"Predicted Class: {result['class_names'][result['predicted_class']]}")
        print(f"Confidence: {result['confidence']*100:.2f}%")

        if show_all_probabilities:
            print(f"\nAll Class Probabilities:")
            print(f"{'-'*60}")
            for idx, (class_name, prob) in enumerate(zip(result['class_names'], result['probabilities'])):
                marker = "→" if idx == result['predicted_class'] else " "
                print(f"{marker} {class_name:<25} {prob*100:6.2f}%")

        print(f"{'='*60}\n")

        return result

    def get_model_info(self):
        """Get information about the loaded model"""
        print(f"\n{'='*60}")
        print(f"MODEL INFORMATION")
        print(f"{'='*60}")
        print(f"Model Name: {self.metadata['model_name']}")
        print(f"Architecture: {self.metadata['model_class']}")
        print(f"Number of Classes: {self.metadata['num_classes']}")
        print(f"Training Date: {self.metadata['training_date']}")
        print(f"\nTraining Performance:")
        print(f"  Best Validation Loss: {self.metadata['best_val_loss']:.4f}")
        print(f"  Best Validation Accuracy: {self.metadata['best_val_acc']*100:.2f}%")
        print(f"  Final Training Accuracy: {self.metadata['final_train_acc']*100:.2f}%")
        print(f"\nTraining Configuration:")
        print(f"  Total Epochs: {self.metadata['total_epochs_trained']}")
        print(f"  Batch Size: {self.metadata['batch_size']}")
        print(f"  Learning Rate: {self.metadata['learning_rate']}")
        print(f"  Total Training Time: {self._format_time(self.metadata['total_training_time'])}")
        print(f"  Device Used: {self.metadata['device']}")
        print(f"{'='*60}\n")

        return self.metadata

    def _format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


def load_model_for_prediction(model_package_dir):
    """
    Convenience function to load a trained model for predictions

    Args:
        model_package_dir: Path to the saved model package directory

    Returns:
        ModelLoader: Loaded model ready for predictions
    """
    return ModelLoader(model_package_dir)


def compare_models_prediction(model_dirs, image_path):
    """
    Compare predictions from multiple trained models on the same image

    Args:
        model_dirs: List of model package directories
        image_path: Path to the image to predict
    """
    print(f"\n{'='*80}")
    print(f"COMPARING MULTIPLE MODELS")
    print(f"{'='*80}")
    print(f"Image: {Path(image_path).name}\n")

    results = []
    for model_dir in model_dirs:
        try:
            loader = ModelLoader(model_dir)
            result = loader.predict_image(image_path)
            result['model_name'] = loader.metadata['model_name']
            result['model_accuracy'] = loader.metadata['best_val_acc']
            results.append(result)
        except Exception as e:
            print(f"Error loading model from {model_dir}: {e}")

    # Display comparison
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    print(f"{'Model':<15} {'Prediction':<25} {'Confidence':<12} {'Val Accuracy':<15}")
    print(f"{'-'*80}")

    for result in results:
        pred_class = class_names[result['predicted_class']]
        confidence = result['confidence'] * 100
        accuracy = result['model_accuracy'] * 100
        print(f"{result['model_name']:<15} {pred_class:<25} {confidence:6.2f}% {accuracy:>12.2f}%")

    print(f"{'='*80}\n")

    return results


# Example usage
if __name__ == "__main__":
    print("Model Loader Module - Example Usage")
    print("="*60)

    # Example 1: Load a single model and make predictions
    print("\nExample 1: Loading a trained model")
    print("-"*60)
    print("""
# Load a trained model
from modules.model_loader import load_model_for_prediction

model_loader = load_model_for_prediction('path/to/model/package')

# Make prediction on a single image
result = model_loader.predict_with_details('path/to/image.png')

# Get model information
model_loader.get_model_info()
""")

    # Example 2: Batch predictions
    print("\nExample 2: Batch predictions")
    print("-"*60)
    print("""
# Predict on multiple images
image_paths = ['image1.png', 'image2.png', 'image3.png']
results = model_loader.predict_batch(image_paths)

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['class_names'][result['predicted_class']]}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
""")

    # Example 3: Compare multiple models
    print("\nExample 3: Compare predictions from multiple models")
    print("-"*60)
    print("""
from modules.model_loader import compare_models_prediction

model_dirs = [
    'models/VGG16_20241110_120000',
    'models/ResNet50_20241110_130000',
    'models/InceptionV3_20241110_140000'
]

results = compare_models_prediction(model_dirs, 'test_image.png')
""")
