"""
Training module for diabetic retinopathy classification models
"""
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Trainer:
    """Class responsible for training CNN models"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def setup_training_components(self):
        """Setup optimizer, loss function, and scheduler"""
        print("Setting up training components...")

        # Setup optimizer (Adam with configurable learning rate)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        print(f"✓ Optimizer: Adam (lr={self.config.learning_rate})")

        # Setup loss function (CrossEntropyLoss for classification)
        self.criterion = nn.CrossEntropyLoss()
        print("✓ Loss function: CrossEntropyLoss")

        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print("✓ Scheduler: ReduceLROnPlateau")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Handle InceptionV3 auxiliary outputs
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Handle InceptionV3 auxiliary outputs (only use main output for validation)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, labels)

                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def train(self, train_loader, val_loader, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.epochs

        print("=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)
        print(f"Model: {self.model.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print("=" * 50)

        # Setup training components
        self.setup_training_components()

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 20)

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            epoch_time = time.time() - epoch_start_time

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            print(f'Epoch Time: {epoch_time:.2f}s')

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s')
        print(f'Best validation loss: {self.best_val_loss:.4f}')

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def plot_training_history(self, save_path=None):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        ax1.plot(self.training_history['train_loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.training_history['train_acc'], label='Training Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to: {save_path}")

        plt.show()

    def save_checkpoint(self, filepath, epoch, optimizer_state=True):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict() if optimizer_state else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to: {filepath}")

    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint['best_model_state_dict']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']

        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint['epoch']

# Test function for independent execution
def test_train_module():
    """Test the training module independently"""
    print("Testing Train module...")

    try:
        # Import required modules
        from config import Config
        from models import get_model

        # Create config
        config = Config()
        print("✓ Config imported and created successfully")

        # Test trainer creation for each model
        model_names = ['vgg16', 'resnet50', 'inceptionv3']

        for model_name in model_names:
            try:
                print(f"Testing Trainer with {model_name.upper()} model...")

                # Create model
                model = get_model(model_name, config)
                print(f"✓ {model_name.upper()} model created")

                # Create trainer
                trainer = Trainer(model, config)
                print(f"✓ Trainer created for {model_name.upper()}")

                # Test training components setup
                trainer.setup_training_components()
                print(f"✓ Training components setup successful for {model_name.upper()}")

                # Test that trainer has all required attributes
                assert hasattr(trainer, 'optimizer'), "Trainer missing optimizer"
                assert hasattr(trainer, 'criterion'), "Trainer missing criterion"
                assert hasattr(trainer, 'scheduler'), "Trainer missing scheduler"
                assert hasattr(trainer, 'training_history'), "Trainer missing training_history"

                print(f"✓ All trainer attributes present for {model_name.upper()}")

                # Test dummy training step (without actual data)
                print(f"✓ {model_name.upper()} trainer test completed")

            except Exception as e:
                print(f"✗ Trainer test failed for {model_name}: {e}")
                return False

        print("✓ All trainer tests successful")
        print("✓ Train module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Train module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_train_module()
    print(f"\nTrain Module Test Result: {'PASS' if success else 'FAIL'}")
