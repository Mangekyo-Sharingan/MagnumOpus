"""
Main orchestration file for diabetic retinopathy classification project
"""
from modules import Config, DataLoader, ModelFactory, Trainer, Evaluator, Utils, DeviceManager
from modules.hyperparameter_tuner import HyperparameterTuner, QuickTuner
import torch
import numpy as np

class DiabetitcRetinopathyClassifier:
    """Main class that orchestrates the entire classification pipeline"""

    def __init__(self):
        self.config = Config()
        self.data_loader = None
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}
        self.best_hyperparams = {}  # Store best params for each model

        # Set up device and reproducibility
        self.device = DeviceManager.get_device()
        Utils.set_seed(self.config.random_state)

    def setup_project(self):
        """Initialize project setup"""
        print("Setting up diabetic retinopathy classification project...")
        self.config.create_directories()
        self.data_loader = DataLoader(self.config)

    def prepare_data(self):
        """Load and prepare data for training"""
        print("Loading and preparing data...")
        self.data_loader.load_data()
        # Additional data preparation steps

    def select_models(self):
        """Allow user to select which models to train"""
        print("\n" + "=" * 100)
        print(" " * 35 + "MODEL SELECTION")
        print("=" * 100)
        print("\nAvailable models for training:")
        print("-" * 100)

        for idx, model_name in enumerate(self.config.models, 1):
            config_info = self.config.model_configs.get(model_name, {})
            image_size = config_info.get('image_size', 'Unknown')
            pipeline = config_info.get('pipeline', 'Unknown')
            print(f"  {idx}. {model_name.upper():<15} | Image Size: {str(image_size):<12} | Pipeline: {pipeline}")

        print("-" * 100)
        print("\nSelection Options:")
        print("  • Enter model numbers separated by commas (e.g., 1,3)")
        print("  • Enter 'all' to train all models")
        print("  • Enter individual numbers for specific models")
        print("-" * 100)

        while True:
            user_input = input("\n➤ Select models to train: ").strip().lower()

            if user_input == 'all':
                selected_models = self.config.models.copy()
                print(f"\n✓ Selected ALL models: {', '.join(selected_models)}")
                break
            elif user_input:
                try:
                    # Parse comma-separated numbers
                    indices = [int(x.strip()) for x in user_input.split(',')]
                    selected_models = []

                    for idx in indices:
                        if 1 <= idx <= len(self.config.models):
                            selected_models.append(self.config.models[idx - 1])
                        else:
                            print(f"⚠️  Invalid selection: {idx}. Please choose from 1-{len(self.config.models)}")
                            selected_models = None
                            break

                    if selected_models:
                        print(f"\n✓ Selected models: {', '.join(selected_models)}")
                        break
                except ValueError:
                    print("⚠️  Invalid input format. Please enter numbers separated by commas or 'all'")
            else:
                print("⚠️  No selection made. Please select at least one model.")

        print("=" * 100 + "\n")
        return selected_models

    def initialize_models(self, selected_models=None):
        """Initialize selected CNN models"""
        if selected_models is None:
            selected_models = self.config.models

        print("\n" + "=" * 100)
        print(" " * 35 + "MODEL INITIALIZATION")
        print("=" * 100)
        print(f"\nInitializing {len(selected_models)} model(s)...")
        print("-" * 100)

        for idx, model_name in enumerate(selected_models, 1):
            print(f"\n[{idx}/{len(selected_models)}] Creating {model_name.upper()} model...")
            self.models[model_name] = ModelFactory.create_model(model_name, self.config)

            # Print model information
            param_info = Utils.count_parameters(self.models[model_name])
            print(f"  ✓ {model_name.upper()} initialized")
            print(f"    • Total parameters: {param_info['total_parameters']:,}")
            print(f"    • Trainable parameters: {param_info['trainable_parameters']:,}")

        print("\n" + "-" * 100)
        print(f"✅ {len(selected_models)} model(s) initialized successfully")
        print("=" * 100 + "\n")

    def tune_hyperparameters(self, model_name):
        """
        Tune hyperparameters for a specific model before training
        Each model gets its own tuning since different architectures need different params!

        Args:
            model_name: Name of the model to tune
        """
        print("\n" + "=" * 100)
        print(f" " * 25 + f"HYPERPARAMETER TUNING FOR {model_name.upper()}")
        print("=" * 100)

        # Let user choose tuning mode
        print(f"\n🎯 Each model needs optimal hyperparameters for best performance!")
        print(f"   {model_name.upper()} will be tuned before training.\n")
        print(f"📊 Select tuning mode:")
        print(f"   1. BALANCED     - 18 combinations, 2000 samples (~15-20 min) [RECOMMENDED]")
        print(f"   2. QUICK        - 18 combinations, 500 samples (~5-10 min)")
        print(f"   3. SUPER-QUICK  - 18 combinations, 50 samples (~1-2 min)")
        print(f"   4. RANDOM       - 18 random combinations, 2000 samples (~15-20 min)")
        print(f"   5. SKIP         - Use default parameters (not recommended)")

        while True:
            response = input("\n➤ Select tuning mode (1-5): ").strip()

            if response == '1':
                tuning_mode = 'balanced'
                n_samples = 2000
                break
            elif response == '2':
                tuning_mode = 'quick'
                n_samples = 500
                break
            elif response == '3':
                tuning_mode = 'super-quick'
                n_samples = 50
                break
            elif response == '4':
                tuning_mode = 'random'
                n_samples = 2000
                break
            elif response == '5':
                print(f"\n⏭️  Skipping hyperparameter tuning for {model_name.upper()}")
                print(f"📋 Using default parameters:")
                print(f"   • Batch Size: {self.config.batch_size}")
                print(f"   • Learning Rate: {self.config.learning_rate}")
                return None
            else:
                print(f"⚠️  Invalid selection. Please enter 1-5.")

        # Get tuning dataset
        print(f"\n📊 Preparing tuning dataset...")
        train_loader, val_loader = self.data_loader.create_data_loaders(model_name)
        tuning_dataset = train_loader.dataset

        # Model factory function
        def model_factory():
            return ModelFactory.create_model(model_name, self.config)

        # Configure tuning - ALL MODES USE 18 COMBINATIONS
        param_grid = {
            'batch_size': [16, 32, 64],           # 3 values
            'epochs': [25, 50, 75, 100],                   # 2 values
            'learning_rate': [0.001, 0.005, 0.01] # 3 values
        }
        # 3 × 2 × 3 = 18 combinations for all modes

        if tuning_mode == 'random':
            print(f"\n🎲 Running RANDOM search (18 random combinations, {n_samples} samples)...")
            tuner = QuickTuner(model_factory, self.config, tuning_dataset, self.config.random_state)
            param_ranges = {
                'batch_size': [16, 32, 64],
                'epochs': [10, 15, 20, 25, 30],
                'learning_rate': (0.001, 0.01)
            }
            best_params = tuner.random_search(param_ranges, n_iterations=18, n_samples=n_samples)
            self.best_hyperparams[model_name] = best_params
            return best_params

        else:  # balanced, quick, or thorough (all use grid search)
            mode_name = tuning_mode.upper()
            print(f"\n🔍 Running {mode_name} grid search (18 combinations, {n_samples} samples)...")

            # Create tuner and run grid search
            tuner = HyperparameterTuner(model_factory, self.config, tuning_dataset, self.config.random_state)
            best_params = tuner.grid_search(param_grid=param_grid, n_samples=n_samples, save_results=True)

        if best_params:
            self.best_hyperparams[model_name] = best_params

            # Update config for this model
            self.config.batch_size = best_params['batch_size']
            self.config.learning_rate = best_params['learning_rate']

            print(f"\n✅ Tuning complete! Updated configuration for {model_name.upper()}:")
            print(f"   • Batch Size: {self.config.batch_size}")
            print(f"   • Learning Rate: {self.config.learning_rate}")

        print("=" * 100 + "\n")

        return best_params

    def train_models(self):
        """Train all models with pause between each model"""
        print("Training models...")
        total_models = len(self.models)

        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"\n{'='*100}")
            print(f"{'='*100}")
            print(f"  TRAINING MODEL {idx}/{total_models}: {model_name.upper()}")
            print(f"{'='*100}")
            print(f"{'='*100}\n")

            # HYPERPARAMETER TUNING - Each model gets optimized individually!
            best_params = self.tune_hyperparameters(model_name)

            # Create data loaders for this model (with potentially updated batch size)
            train_loader, val_loader = self.data_loader.create_data_loaders(model_name)

            # Initialize trainer (with potentially updated learning rate)
            trainer = Trainer(model, self.config)
            self.trainers[model_name] = trainer

            # Train the model
            trainer.train_model(train_loader, val_loader)

            # Save complete model package (includes everything needed for future predictions)
            models_dir = self.config.base_dir / "models"
            model_package_dir = trainer.save_complete_model(models_dir, model_name)

            print(f"\n{'='*100}")
            print(f"✅ {model_name.upper()} TRAINING AND SAVING COMPLETED!")
            print(f"{'='*100}")
            print(f"📦 Model package location: {model_package_dir}")
            if best_params:
                print(f"🎯 Best hyperparameters used:")
                print(f"   • Batch Size: {best_params['batch_size']}")
                print(f"   • Learning Rate: {best_params['learning_rate']}")
            print(f"{'='*100}\n")

            # Pause and wait for user input before next model (unless it's the last model)
            if idx < total_models:
                print(f"\n{'─'*100}")
                print(f"⏸️  PAUSED: {model_name.upper()} training complete.")
                print(f"{'─'*100}")
                print(f"📊 Models completed: {idx}/{total_models}")
                print(f"📋 Remaining models: {', '.join(list(self.models.keys())[idx:])}")
                print(f"{'─'*100}")

                user_input = input(f"\n▶️  Press ENTER to continue training {list(self.models.keys())[idx]} or type 'skip' to end training: ")

                if user_input.lower().strip() == 'skip':
                    print(f"\n⏹️  Training stopped by user. {total_models - idx} model(s) remaining.")
                    break
                else:
                    print(f"\n▶️  Resuming training...\n")

        print(f"\n{'='*100}")
        print(f"🎉 ALL TRAINING SESSIONS COMPLETED!")
        print(f"{'='*100}")
        print(f"✓ Total models trained: {len(self.trainers)}/{total_models}")
        print(f"✓ All model packages saved in: {self.config.base_dir / 'models'}")
        print(f"{'='*100}\n")

    def evaluate_models(self):
        """Evaluate all trained models"""
        print("Evaluating models...")
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")

            # Create test data loader
            test_loader = self.data_loader.create_test_loader(model_name)

            # Initialize evaluator
            evaluator = Evaluator(model, self.config)
            self.evaluators[model_name] = evaluator

            # Evaluate the model
            evaluator.evaluate_model(test_loader)

            # Calculate and display metrics
            metrics = evaluator.calculate_metrics()
            self.results[model_name] = metrics

            print(f"{model_name} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
            print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
            print(f"  Weighted Recall: {metrics['recall_weighted']:.4f}")

            # Generate classification report
            evaluator.generate_classification_report()

            # Plot confusion matrix
            cm_path = self.config.base_dir / "results" / f"{model_name}_confusion_matrix.png"
            evaluator.plot_confusion_matrix(save_path=cm_path)

            # Save evaluation results
            results_path = self.config.base_dir / "results" / f"{model_name}_evaluation_results.npy"
            evaluator.save_results(results_path)

            print(f"{model_name} evaluation completed!")

    def compare_results(self):
        """Compare results from all models"""
        print("\nComparing model results...")
        print("=" * 60)

        # Display comparison table
        print(f"{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 60)

        for model_name, metrics in self.results.items():
            print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['f1_weighted']:<10.4f} "
                  f"{metrics['precision_weighted']:<12.4f} {metrics['recall_weighted']:<10.4f}")

        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model]['accuracy']

        print(f"\nBest performing model: {best_model} (Accuracy: {best_accuracy:.4f})")

        # Create comparison visualization
        from modules.utils import Visualizer
        visualizer = Visualizer()

        comparison_path = self.config.base_dir / "results" / "model_comparison.png"
        visualizer.plot_model_comparison(self.results, metric='accuracy', save_path=comparison_path)

        # Save comparison results
        comparison_data = {
            'results': self.results,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'timestamp': Utils.get_timestamp()
        }

        comparison_file = self.config.base_dir / "results" / "model_comparison_results.json"
        Utils.save_json(comparison_data, comparison_file)

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "=" * 100)
        print(" " * 25 + "DIABETIC RETINOPATHY CLASSIFICATION PIPELINE")
        print("=" * 100)

        try:
            # Setup
            self.setup_project()

            # Data preparation
            self.prepare_data()

            # Model selection (NEW)
            selected_models = self.select_models()

            # Model initialization with selected models
            self.initialize_models(selected_models)

            # Training
            self.train_models()

            # Evaluation
            self.evaluate_models()

            # Comparison
            self.compare_results()

            print("\n" + "=" * 70)
            print("Pipeline completed successfully!")
            print(f"Results saved in: {self.config.base_dir / 'results'}")
            print(f"Models saved in: {self.config.base_dir / 'models'}")

        except Exception as e:
            print(f"\nError occurred during pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the classification pipeline"""
    classifier = DiabetitcRetinopathyClassifier()
    classifier.run_full_pipeline()

if __name__ == "__main__":
    main()
