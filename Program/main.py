#Main orchestration file for project
import os

#Set ROCm preallocation BEFORE importing torch
os.environ.setdefault('PYTORCH_MIOPEN_PREALLOC_MB', '14336')  # Pre-allocate 14GB GPU memory
os.environ['PYTORCH_MIOPEN_PREALLOC_MB'] = '14336'

#Avoid repeated initialization in worker processes
_TORCH_INITIALIZED = False

from modules import Config, DataLoader, ModelFactory, Trainer, Evaluator, Utils, DeviceManager, ResourceMonitor
import torch

def initialize_torch_settings():
    #Initialize PyTorch settings once at startup.
    global _TORCH_INITIALIZED
    if _TORCH_INITIALIZED:
        return

    #GPU Optimization Settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #Enable mixed precision matmul
    try:
        torch.set_float32_matmul_precision('medium')
        print("Float32 matmul precision set to medium")
    except Exception as e:
        print(f"Could not set matmul precision: {e}")

    _TORCH_INITIALIZED = True

class DiabetitcRetinopathyClassifier:
    #Main class that orchestrates the entire pipeline

    def __init__(self):
        self.config = Config()

        if self.config.enable_miopen_fix:
            print("\n Applying MIOpen fixes before initializing the device")
            DeviceManager.apply_miopen_fixes(self.config)
        else:
            print("\n MIOpen fixes disabled via configuration")

        self.data_loader = None
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}
        self.best_hyperparams = {}  # Store best params for each model

        # Set up device and reproducibility
        self.device = DeviceManager.get_device()
        Utils.set_seed(self.config.random_state)

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(
            log_dir=self.config.base_dir / 'logs' / 'resources',
            interval=1.0,  # 1 second base sampling
            adaptive=True  # Enable adaptive sampling
        )

    def setup_project(self):
        #Initialize project setup
        print("Setting up project")
        self.config.create_directories()
        self.data_loader = DataLoader(self.config)

    def prepare_data(self):
        #Load and prepare data for training
        print("Loading and preparing data")
        self.data_loader.load_data()
        # Additional data preparation steps

    def select_models(self):
        #Allow user to select which models to train
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
        print("  Enter model numbers separated by commas (e.g., 1,3)")
        print("  Enter 'all' to train all models")
        print("  Enter individual numbers for specific models")
        print("-" * 100)

        while True:
            user_input = input("\n> Select models to train: ").strip().lower()

            if user_input == 'all':
                selected_models = self.config.models.copy()
                print(f"\nSelected ALL models: {', '.join(selected_models)}")
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
                            print(f"Invalid selection: {idx}. Choose from 1-{len(self.config.models)}")
                            selected_models = None
                            break

                    if selected_models:
                        print(f"\nSelected models: {', '.join(selected_models)}")
                        break
                except ValueError:
                    print("Invalid input format. Enter numbers separated by commas or 'all'")
            else:
                print("No selection made. Select at least one model.")

        print("=" * 100 + "\n")
        return selected_models

    def initialize_models(self, selected_models=None):
        #Initialize selected CNN models
        if selected_models is None:
            selected_models = self.config.models

        print("\n" + "=" * 100)
        print(" " * 35 + "MODEL INITIALIZATION")
        print("=" * 100)
        print(f"\nInitializing {len(selected_models)} model(s)")
        print("-" * 100)

        for idx, model_name in enumerate(selected_models, 1):
            print(f"\n[{idx}/{len(selected_models)}] Creating {model_name.upper()} model")
            self.models[model_name] = ModelFactory.create_model(model_name, self.config)

            # Print model information
            param_info = Utils.count_parameters(self.models[model_name])
            print(f"{model_name.upper()} initialized")
            print(f"Total parameters: {param_info['total_parameters']:,}")
            print(f"Trainable parameters: {param_info['trainable_parameters']:,}")

        print("\n" + "-" * 100)
        print(f"{len(selected_models)} model initialized successfully")
        print("=" * 100 + "\n")

    def train_models(self):
        #Train all models with pause between each model
        print("Training models")
        total_models = len(self.models)

        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"\n{'='*100}")
            print(f"  TRAINING MODEL {idx}/{total_models}: {model_name.upper()}")
            print(f"{'='*100}\n")

            # Get model hyperparameters from config
            model_config = self.config.get_model_config(model_name)
            batch_size = model_config['batch_size']
            learning_rate = model_config['learning_rate']
            epochs = model_config['epochs']

            print("Using model hyperparameters from config.py:")
            print(f"  - Batch Size: {batch_size}")
            print(f"  - Learning Rate: {learning_rate}")
            print(f"  - Epochs: {epochs}")

            # Create data loaders for this model
            # Note: The batch size from the config is used here
            train_loader, val_loader = self.data_loader.create_data_loaders(model_name, batch_size=batch_size)

            # Initialize trainer with resource monitor
            trainer = Trainer(model, self.config, model_name=model_name, resource_monitor=self.resource_monitor)
            self.trainers[model_name] = trainer

            # Train the model
            trainer.train_model(train_loader, val_loader, num_epochs=epochs)

            # Save complete model package
            models_dir = self.config.base_dir / "models"
            model_package_dir = trainer.save_complete_model(models_dir, model_name)

            print(f"\n{'='*100}")
            print(f"{model_name.upper()} TRAINING AND SAVING COMPLETED!")
            print(f"Model package location: {model_package_dir}")
            print(f"{'='*100}\n")

            if idx < total_models:
                input(f"\n>Press ENTER to continue training the next model")

        print(f"\n{'='*100}")
        print(f"ALL TRAINING SESSIONS COMPLETED!")
        print(f"{'='*100}")
        print(f"Total models trained: {len(self.trainers)}/{total_models}")
        print(f"All model packages saved in: {self.config.base_dir / 'models'}")
        print(f"{'='*100}\n")

    def evaluate_models(self):
        #Evaluate all trained models on the test set.
        print("Evaluating models on the test set...")
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}")

            # Get the test loader created during the data split
            test_loader = self.data_loader.get_test_loader()
            if not test_loader:
                print(f"Skipping evaluation for {model_name}, no test loader found.")
                continue

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
            report_path = self.config.base_dir / "results" / f"{model_name}_classification_report.txt"
            evaluator.generate_classification_report(save_path=report_path)

            # Plot confusion matrix
            cm_path = self.config.base_dir / "results" / f"{model_name}_confusion_matrix.png"
            evaluator.plot_confusion_matrix(save_path=cm_path)

            # Save evaluation results
            results_path = self.config.base_dir / "results" / f"{model_name}_evaluation_results.npy"
            evaluator.save_results(results_path)
            print(f"{model_name} evaluation completed!")

    def compare_results(self):
        #Compare results from all models
        print("\nComparing model results")
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
        #Run the complete pipeline
        print("\n" + "=" * 100)
        print(" " * 25 + "DIABETIC RETINOPATHY CLASSIFICATION PIPELINE")
        print("=" * 100)

        # Start resource monitor
        self.resource_monitor.start()
        try:
            # Setup
            self.setup_project()
            # Data preparation
            self.prepare_data()
            # Model selection
            selected_models = self.select_models()
            # Model initialization
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
        finally:
            # Stop monitor even if error occurs
            print("\n" + "=" * 100)
            print("Stopping resource monitor")
            self.resource_monitor.stop()

def main():
    #Main function to run the classification pipeline
    initialize_torch_settings()
    classifier = DiabetitcRetinopathyClassifier()
    classifier.run_full_pipeline()

if __name__ == '__main__':
    main()
