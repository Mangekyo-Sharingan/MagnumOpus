"""
Training module for diabetic retinopathy classification models
"""
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import gc

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
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': [],
            'samples_per_second': [],
            'gpu_memory_used': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Progress tracking
        self.epoch_start_time = None
        self.total_start_time = None
        self.epoch_times = []
        self.total_samples_processed = 0
        
    def _get_memory_usage(self):
        """Get current memory usage (GPU if available, else RAM)"""
        try:
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                return memory_gb, max_memory_gb
            else:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3, 0.0
        except:
            return 0.0, 0.0

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
            patience=5
        )
        print("✓ Scheduler: ReduceLROnPlateau")

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
    
    def _estimate_remaining_time(self, current_epoch, total_epochs):
        """Estimate remaining training time based on average epoch time"""
        if not self.epoch_times:
            return "Calculating..."
        
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        remaining_seconds = avg_epoch_time * remaining_epochs
        
        return self._format_time(remaining_seconds)
    
    def _print_progress_bar(self, current, total, prefix='', suffix='', length=40, fill='█'):
        """Print a progress bar"""
        percent = 100 * (current / float(total))
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
        if current == total:
            print()

    def _print_training_header(self, num_epochs, train_loader, val_loader):
        """Print a comprehensive training header with all details"""
        print("\n" + "=" * 100)
        print(" " * 35 + "TRAINING INITIALIZATION")
        print("=" * 100)
        print(f"📋 Model Configuration:")
        print(f"   • Model Name      : {self.model.model_name}")
        print(f"   • Device          : {self.device}")
        if torch.cuda.is_available():
            print(f"   • GPU Name        : {torch.cuda.get_device_name(0)}")
            print(f"   • GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1024**6:.2f} GB")
        print(f"\n Training Configuration:")
        print(f"   • Total Epochs    : {num_epochs}")
        print(f"   • Batch Size      : {self.config.batch_size}")
        print(f"   • Learning Rate   : {self.config.learning_rate}")
        print(f"   • Train Batches   : {len(train_loader)}")
        print(f"   • Val Batches     : {len(val_loader)}")
        print(f"   • Train Samples   : ~{len(train_loader) * self.config.batch_size}")
        print(f"   • Val Samples     : ~{len(val_loader) * self.config.batch_size}")
        print("=" * 100 + "\n")

    def train_epoch(self, train_loader, epoch_num=None, total_epochs=None):
        """Train for one epoch with detailed progress tracking"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        batch_times = []
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            
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
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress bar every 10 batches or at the end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                avg_batch_time = np.mean(batch_times[-10:])  # Last 10 batches
                
                # Estimate time remaining for this epoch
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * avg_batch_time
                eta_str = self._format_time(eta_seconds)
                
                # Memory usage
                gpu_memory, max_gpu_memory = self._get_memory_usage()
                memory_str = f" | GPU Mem: {gpu_memory:.2f}GB / {max_gpu_memory:.2f}GB" if gpu_memory > 0 else ""

                suffix = f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f} | ETA: {eta_str}{memory_str}"
                if epoch_num and total_epochs:
                    prefix = f"Epoch {epoch_num}/{total_epochs} - Training"
                else:
                    prefix = "Training"
                
                self._print_progress_bar(batch_idx + 1, total_batches, 
                                        prefix=prefix, suffix=suffix, length=30)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        # Calculate and log samples per second
        samples_per_second = total_samples / np.sum(batch_times)
        self.training_history['samples_per_second'].append(samples_per_second)

        return epoch_loss, epoch_acc.item()

    def validate_epoch(self, val_loader, epoch_num=None, total_epochs=None):
        """Validate for one epoch with progress tracking"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        total_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
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
                
                # Progress bar
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    current_loss = running_loss / total_samples
                    current_acc = running_corrects.double() / total_samples
                    suffix = f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}"
                    
                    if epoch_num and total_epochs:
                        prefix = f"Epoch {epoch_num}/{total_epochs} - Validation"
                    else:
                        prefix = "Validation"
                    
                    self._print_progress_bar(batch_idx + 1, total_batches,
                                            prefix=prefix, suffix=suffix, length=30)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def train(self, train_loader, val_loader, num_epochs=None):
        """Main training loop with comprehensive progress tracking"""
        if num_epochs is None:
            num_epochs = self.config.epochs

        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Model: {self.model.model_name}")
        print(f"Device: {self.device}")
        print(f"Total Epochs: {num_epochs}")
        print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 80)

        # Setup training components
        self.setup_training_components()

        # Print comprehensive training header
        self._print_training_header(num_epochs, train_loader, val_loader)

        self.total_start_time = time.time()
        best_epoch = 0

        for epoch in range(num_epochs):
            self.epoch_start_time = time.time()

            print(f'\n{"="*80}')
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'{"="*80}')

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1, num_epochs)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader, epoch+1, num_epochs)

            # Calculate epoch time
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            improvement = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                improvement = " ⭐ NEW BEST!"

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['learning_rates'].append(current_lr)

            # Calculate statistics
            avg_epoch_time = np.mean(self.epoch_times)
            eta = self._estimate_remaining_time(epoch + 1, num_epochs)
            total_elapsed = time.time() - self.total_start_time
            
            # Print epoch summary
            print(f'\n{"─"*80}')
            print(f'📊 EPOCH {epoch+1}/{num_epochs} SUMMARY{improvement}')
            print(f'{"─"*80}')
            print(f'Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
            print(f'Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}')
            print(f'{"─"*80}')
            print(f'⏱️  Epoch Time: {self._format_time(epoch_time)} | Avg: {self._format_time(avg_epoch_time)}')
            print(f'📈 Learning Rate: {current_lr:.6f}')
            print(f'⏳ Elapsed: {self._format_time(total_elapsed)} | ETA: {eta}')
            print(f'🏆 Best Val Loss: {self.best_val_loss:.4f} (Epoch {best_epoch})')
            print(f'{"─"*80}')

        total_time = time.time() - self.total_start_time
        
        print(f'\n{"="*80}')
        print(f'✅ TRAINING COMPLETED!')
        print(f'{"="*80}')
        print(f'Total Time: {self._format_time(total_time)}')
        print(f'Average Epoch Time: {self._format_time(np.mean(self.epoch_times))}')
        print(f'Best Validation Loss: {self.best_val_loss:.4f} (Epoch {best_epoch})')
        print(f'Final Training Accuracy: {train_acc:.4f}')
        print(f'Final Validation Accuracy: {val_acc:.4f}')
        print(f'{"="*80}\n')

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✓ Loaded best model from epoch {best_epoch}")

    def train_model(self, train_loader, val_loader, num_epochs=None):
        """Alias for train() method for compatibility with main.py"""
        return self.train(train_loader, val_loader, num_epochs)

    def save_training_history(self, filepath):
        """Save training history to file"""
        import numpy as np
        np.save(filepath, self.training_history)
        print(f"Training history saved to: {filepath}")

    def plot_training_metrics(self, save_path=None):
        """Alias for plot_training_history for compatibility with main.py"""
        return self.plot_training_history(save_path)

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

    def save_complete_model(self, save_dir, model_name=None):
        """
        Save complete model package for future predictions
        Includes: model weights, architecture info, training config, and metadata
        """
        from pathlib import Path
        import json
        from datetime import datetime

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            model_name = self.model.model_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model-specific directory
        model_dir = save_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"💾 SAVING COMPLETE MODEL PACKAGE: {model_name}")
        print(f"{'='*80}")

        # 1. Save model state (weights + architecture)
        model_weights_path = model_dir / f"{model_name}_model.pth"
        torch.save({
            'model_state_dict': self.best_model_state if self.best_model_state else self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_name': model_name,
            'num_classes': self.config.num_classes,
            'input_size': getattr(self.config, 'input_size', (224, 224)),
        }, model_weights_path)
        print(f"✓ Model weights saved: {model_weights_path.name}")

        # 2. Save training history
        history_path = model_dir / f"{model_name}_training_history.npy"
        np.save(history_path, self.training_history)
        print(f"✓ Training history saved: {history_path.name}")

        # 3. Save optimizer state (for potential resume)
        if self.optimizer:
            optimizer_path = model_dir / f"{model_name}_optimizer.pth"
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, optimizer_path)
            print(f"✓ Optimizer state saved: {optimizer_path.name}")

        # 4. Save configuration and metadata
        metadata = {
            'model_name': model_name,
            'model_class': self.model.__class__.__name__,
            'num_classes': self.config.num_classes,
            'training_date': timestamp,
            'total_epochs_trained': len(self.training_history['train_loss']),
            'best_val_loss': float(self.best_val_loss),
            'best_val_acc': float(max(self.training_history['val_acc'])) if self.training_history['val_acc'] else 0.0,
            'final_train_loss': float(self.training_history['train_loss'][-1]) if self.training_history['train_loss'] else 0.0,
            'final_train_acc': float(self.training_history['train_acc'][-1]) if self.training_history['train_acc'] else 0.0,
            'final_val_loss': float(self.training_history['val_loss'][-1]) if self.training_history['val_loss'] else 0.0,
            'final_val_acc': float(self.training_history['val_acc'][-1]) if self.training_history['val_acc'] else 0.0,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'device': str(self.device),
            'total_training_time': sum(self.epoch_times) if self.epoch_times else 0.0,
            'average_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
        }

        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved: {metadata_path.name}")

        # 5. Save training plots
        plot_path = model_dir / f"{model_name}_training_plots.png"
        self.plot_training_history(save_path=plot_path)
        print(f"✓ Training plots saved: {plot_path.name}")

        # 6. Create README with instructions
        readme_content = f"""# {model_name} - Trained Model Package

## Model Information
- **Model Architecture**: {self.model.__class__.__name__}
- **Training Date**: {timestamp}
- **Number of Classes**: {self.config.num_classes}
- **Best Validation Loss**: {self.best_val_loss:.4f}
- **Best Validation Accuracy**: {max(self.training_history['val_acc']) if self.training_history['val_acc'] else 0.0:.4f}

## Training Summary
- **Total Epochs**: {len(self.training_history['train_loss'])}
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Total Training Time**: {self._format_time(sum(self.epoch_times)) if self.epoch_times else '0s'}
- **Device**: {self.device}

## Files in this Package
1. `{model_name}_model.pth` - Model weights and architecture info
2. `{model_name}_training_history.npy` - Complete training history
3. `{model_name}_optimizer.pth` - Optimizer state (for resume training)
4. `{model_name}_metadata.json` - Training metadata and metrics
5. `{model_name}_training_plots.png` - Training/validation curves
6. `README.md` - This file

## How to Load and Use for Predictions

```python
import torch
from modules import ModelFactory, Config

# Load configuration
config = Config()

# Create model instance
model = ModelFactory.create_model('{model_name.lower()}', config)

# Load trained weights
checkpoint = torch.load('{model_name}_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

## Performance Metrics
- Final Training Loss: {self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0.0:.4f}
- Final Training Accuracy: {self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0.0:.4f}
- Final Validation Loss: {self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0.0:.4f}
- Final Validation Accuracy: {self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0.0:.4f}
"""

        readme_path = model_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ README created: {readme_path.name}")

        print(f"\n{'='*80}")
        print(f"✅ Complete model package saved to: {model_dir}")
        print(f"{'='*80}\n")

        return model_dir

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
                print(f"✗ {model_name.upper()} trainer test failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        print("✓ All trainer tests successful")
        print("✓ Train module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Train module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


class TrainingMonitor:
    """Monitor training progress and provide callbacks"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if training should stop early"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset the monitor"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

