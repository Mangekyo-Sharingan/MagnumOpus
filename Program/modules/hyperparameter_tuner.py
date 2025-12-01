"""
Hyperparameter tuning module for diabetic retinopathy classification models
"""
import time
import json
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split


class HyperparameterTuner:
    """Grid search hyperparameter tuning for CNN models"""

    def __init__(self, model_factory, config, tuning_dataset, random_state=20020315):
        """
        Initialize hyperparameter tuner

        Args:
            model_factory: Function to create fresh model instances
            config: Configuration object
            tuning_dataset: Full dataset to sample from
            random_state: Random seed for reproducibility
        """
        self.model_factory = model_factory
        self.config = config
        self.tuning_dataset = tuning_dataset
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.tuning_results = []
        self.best_params = None
        self.best_score = float('-inf')

    def _create_tuning_subset(self, n_samples=2000):
        """
        Create a fixed subset of data for tuning (uses same random state)

        Args:
            n_samples: Number of samples for tuning

        Returns:
            train_subset, val_subset
        """
        print(f"\n Creating tuning subset with {n_samples} samples...")

        # Get total dataset size
        total_size = len(self.tuning_dataset)

        if n_samples > total_size:
            n_samples = total_size
            print(f"[WARNING]  Requested samples ({n_samples}) exceeds dataset size. Using all {total_size} samples.")

        # Create fixed indices using the same random state
        np.random.seed(self.random_state)
        all_indices = np.arange(total_size)
        np.random.shuffle(all_indices)

        # Select subset
        subset_indices = all_indices[:n_samples]

        # Split into train/val (80/20 split for tuning)
        train_indices, val_indices = train_test_split(
            subset_indices,
            test_size=0.2,
            random_state=self.random_state
        )

        print(f"[OK] Tuning set created: {len(train_indices)} train, {len(val_indices)} validation")

        return train_indices, val_indices

    def _create_data_loaders(self, train_indices, val_indices, batch_size):
        """Create data loaders for tuning"""
        train_subset = Subset(self.tuning_dataset, train_indices)
        val_subset = Subset(self.tuning_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues during tuning
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, val_loader

    def _quick_train_eval(self, model, train_loader, val_loader, lr, epochs):
        """
        Quickly train and evaluate model with given hyperparameters

        Returns:
            val_accuracy: Validation accuracy achieved
        """
        # Setup training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()

        # Training loop with progress
        for epoch in range(epochs):
            running_loss = 0.0
            total_batches = len(train_loader)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # Handle InceptionV3 auxiliary outputs
                if isinstance(outputs, tuple):
                    outputs, aux_outputs = outputs
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Show progress every few batches
                if (batch_idx + 1) % max(1, total_batches // 4) == 0 or (batch_idx + 1) == total_batches:
                    avg_loss = running_loss / (batch_idx + 1)
                    progress = (batch_idx + 1) / total_batches * 100
                    bar_length = 20
                    filled = int(bar_length * (batch_idx + 1) / total_batches)
                    bar = '█' * filled + '─' * (bar_length - filled)
                    print(f'\r      Epoch {epoch+1}/{epochs} |{bar}| {progress:5.1f}% Loss: {avg_loss:.4f}', end='', flush=True)

            print()  # New line after epoch

        # Validation with progress
        model.eval()
        correct = 0
        total = 0
        total_val_batches = len(val_loader)

        print(f'      Validating...', end='', flush=True)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)

                # Handle InceptionV3 auxiliary outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Show validation progress
                if (batch_idx + 1) % max(1, total_val_batches // 2) == 0 or (batch_idx + 1) == total_val_batches:
                    progress = (batch_idx + 1) / total_val_batches * 100
                    print(f'\r      Validating... {progress:5.1f}%', end='', flush=True)

        val_accuracy = 100 * correct / total
        print(f'\r      Validation complete! Accuracy: {val_accuracy:.2f}%')
        return val_accuracy

    def grid_search(self, param_grid=None, n_samples=2000, save_results=True):
        """
        Perform grid search over hyperparameters

        Args:
            param_grid: Dictionary with parameters to search
            n_samples: Number of samples for tuning
            save_results: Whether to save results to file

        Returns:
            best_params: Dictionary with best hyperparameters
        """
        # Default parameter grid (18 combinations - matches main.py)
        if param_grid is None:
            param_grid = {
                'batch_size': [16, 32, 64],           # 3 values
                'epochs': [25, 50, 75],                   # 3 values
                'learning_rate': [0.001, 0.005, 0.01] # 4 values
            }
            # 3 × 2 × 3 = 18 combinations

        print("\n" + "=" * 100)
        print(" " * 30 + "HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 100)
        print(f"\n Search Space:")
        print(f"   - Batch sizes: {param_grid['batch_size']}")
        print(f"   - Epochs (for tuning): {param_grid['epochs']}")
        print(f"   - Learning rates: {param_grid['learning_rate']}")

        # Calculate total combinations
        total_combinations = (len(param_grid['batch_size']) *
                             len(param_grid['epochs']) *
                             len(param_grid['learning_rate']))

        print(f"\n Total combinations to test: {total_combinations}")
        print(f" Tuning on {n_samples} samples (fixed random state: {self.random_state})")
        print("=" * 100)

        # Get user confirmation for large searches
        if total_combinations > 200:
            print(f"\n[WARNING]  WARNING: {total_combinations} combinations will take a long time!")
            response = input("Continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Grid search cancelled.")
                return None

        # Create fixed tuning subset (same for all combinations)
        train_indices, val_indices = self._create_tuning_subset(n_samples)

        # Generate all combinations
        param_combinations = list(itertools.product(
            param_grid['batch_size'],
            param_grid['epochs'],
            param_grid['learning_rate']
        ))

        print(f"\n Starting grid search...")
        start_time = time.time()

        # Test each combination
        for idx, (batch_size, epochs, lr) in enumerate(param_combinations, 1):
            combo_start = time.time()

            # Overall progress bar
            overall_progress = idx / total_combinations * 100
            bar_length = 30
            filled = int(bar_length * idx / total_combinations)
            bar = '█' * filled + '─' * (bar_length - filled)

            print(f"\n{'─'*100}")
            print(f"[{idx}/{total_combinations}] |{bar}| {overall_progress:.1f}% Complete")
            print(f"Testing: Batch Size={batch_size} | Epochs={epochs} | Learning Rate={lr}")
            print(f"{'─'*100}")

            try:
                # Create fresh model
                model = self.model_factory()
                model = model.to(self.device)

                # Create data loaders with current batch size
                train_loader, val_loader = self._create_data_loaders(
                    train_indices, val_indices, batch_size
                )

                # Train and evaluate
                val_accuracy = self._quick_train_eval(model, train_loader, val_loader, lr, epochs)

                # Store results
                result = {
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': lr,
                    'val_accuracy': val_accuracy,
                    'time_taken': time.time() - combo_start
                }
                self.tuning_results.append(result)

                # Update best params
                if val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_params = {
                        'batch_size': batch_size,
                        'epochs': epochs,  # Note: This is just for reference, actual training may use more
                        'learning_rate': lr
                    }
                    print(f" NEW BEST! Accuracy: {val_accuracy:.2f}%")
                else:
                    print(f"Accuracy: {val_accuracy:.2f}%")

                # Clean up
                del model
                del train_loader
                del val_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"[FAIL] Error with combination: {e}")
                continue

            # Progress estimate
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (total_combinations - idx) * avg_time
            print(f"  Progress: {idx}/{total_combinations} | Elapsed: {self._format_time(elapsed)} | ETA: {self._format_time(remaining)}")

        total_time = time.time() - start_time

        # Print results summary
        self._print_results_summary(total_time)

        # Save results
        if save_results:
            self._save_results()

        return self.best_params

    def _format_time(self, seconds):
        """Format seconds into human-readable time"""
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

    def _print_results_summary(self, total_time):
        """Print summary of grid search results"""
        print("\n" + "=" * 100)
        print(" " * 35 + "GRID SEARCH COMPLETE")
        print("=" * 100)

        print(f"\n  Total search time: {self._format_time(total_time)}")
        print(f" Combinations tested: {len(self.tuning_results)}")

        print(f"\n BEST HYPERPARAMETERS:")
        print(f"{'─'*100}")
        print(f"   - Batch Size: {self.best_params['batch_size']}")
        print(f"   - Learning Rate: {self.best_params['learning_rate']}")
        print(f"   - Best Validation Accuracy: {self.best_score:.2f}%")
        print(f"{'─'*100}")

        # Show top 5 combinations
        print(f"\n Top 5 Combinations:")
        sorted_results = sorted(self.tuning_results, key=lambda x: x['val_accuracy'], reverse=True)
        print(f"{'Rank':<6} {'Batch':<8} {'LR':<12} {'Epochs':<10} {'Accuracy':<12}")
        print("─" * 60)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i:<6} {result['batch_size']:<8} {result['learning_rate']:<12.4f} "
                  f"{result['epochs']:<10} {result['val_accuracy']:<12.2f}%")

        print("=" * 100 + "\n")

    def _save_results(self):
        """Save tuning results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.base_dir) / "results" / "hyperparameter_tuning"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = results_dir / f"tuning_results_{timestamp}.json"

        save_data = {
            'timestamp': timestamp,
            'random_state': self.random_state,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.tuning_results,
            'total_combinations': len(self.tuning_results)
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=4)

        print(f" Tuning results saved to: {results_file}")

        # Save best params separately for easy loading
        best_params_file = results_dir / f"best_params_{timestamp}.json"
        with open(best_params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)

        print(f" Best parameters saved to: {best_params_file}")


class QuickTuner:
    """Faster tuning using random search instead of grid search"""

    def __init__(self, model_factory, config, tuning_dataset, random_state=20020315):
        self.model_factory = model_factory
        self.config = config
        self.tuning_dataset = tuning_dataset
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tuning_results = []
        self.best_params = None
        self.best_score = float('-inf')

    def random_search(self, param_ranges, n_iterations=50, n_samples=500):
        """
        Random search: sample random combinations instead of testing all
        Much faster than grid search!

        Args:
            param_ranges: Dictionary with parameter ranges
            n_iterations: Number of random combinations to test
            n_samples: Number of samples for tuning
        """
        print("\n" + "=" * 100)
        print(" " * 30 + "HYPERPARAMETER TUNING - RANDOM SEARCH")
        print("=" * 100)
        print(f"\n Testing {n_iterations} random combinations")
        print(f" Tuning on {n_samples} samples")
        print("=" * 100)

        np.random.seed(self.random_state)

        # Create fixed tuning subset
        tuner = HyperparameterTuner(self.model_factory, self.config,
                                   self.tuning_dataset, self.random_state)
        train_indices, val_indices = tuner._create_tuning_subset(n_samples)

        start_time = time.time()

        for i in range(n_iterations):
            # Sample random hyperparameters
            batch_size = np.random.choice(param_ranges['batch_size'])
            epochs = np.random.choice(param_ranges['epochs'])
            lr = np.random.uniform(param_ranges['learning_rate'][0],
                                  param_ranges['learning_rate'][1])

            print(f"\n{'─'*80}")
            print(f"Iteration {i+1}/{n_iterations}")
            print(f"Batch: {batch_size} | Epochs: {epochs} | LR: {lr:.4f}")

            # Create and train model
            model = self.model_factory().to(self.device)
            train_loader, val_loader = tuner._create_data_loaders(
                train_indices, val_indices, batch_size
            )

            val_accuracy = tuner._quick_train_eval(model, train_loader, val_loader, lr, epochs)

            # Store and check if best
            result = {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': lr,
                'val_accuracy': val_accuracy
            }
            self.tuning_results.append(result)

            if val_accuracy > self.best_score:
                self.best_score = val_accuracy
                self.best_params = result
                print(f" NEW BEST! Accuracy: {val_accuracy:.2f}%")
            else:
                print(f"Accuracy: {val_accuracy:.2f}%")

            # Cleanup
            del model, train_loader, val_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n Best params: {self.best_params}")
        print(f" Best accuracy: {self.best_score:.2f}%")

        return self.best_params

