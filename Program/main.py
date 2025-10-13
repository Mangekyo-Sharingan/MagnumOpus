"""
Main orchestration file for diabetic retinopathy classification project
"""
from modules import Config, DataLoader, ModelFactory, Trainer, Evaluator, Utils, DeviceManager
import torch

class DiabetitcRetinopathyClassifier:
    """Main class that orchestrates the entire classification pipeline"""

    def __init__(self):
        self.config = Config()
        self.data_loader = None
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}

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

    def initialize_models(self):
        """Initialize all CNN models"""
        print("Initializing CNN models...")
        for model_name in self.config.models:
            print(f"Creating {model_name} model...")
            self.models[model_name] = ModelFactory.create_model(model_name, self.config)

            # Print model information
            param_info = Utils.count_parameters(self.models[model_name])
            print(f"{model_name} - Total parameters: {param_info['total_parameters']:,}")
            print(f"{model_name} - Trainable parameters: {param_info['trainable_parameters']:,}")

    def train_models(self):
        """Train all models"""
        print("Training models...")
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            # Create data loaders for this model
            train_loader, val_loader = self.data_loader.create_data_loaders(model_name)

            # Initialize trainer
            trainer = Trainer(model, self.config)
            self.trainers[model_name] = trainer

            # Train the model
            trainer.train_model(train_loader, val_loader)

            # Save training history and plots
            history_path = self.config.base_dir / "results" / f"{model_name}_training_history.npy"
            trainer.save_training_history(history_path)

            plot_path = self.config.base_dir / "results" / f"{model_name}_training_plots.png"
            trainer.plot_training_metrics(save_path=plot_path)

            # Save trained model
            model_path = self.config.base_dir / "models" / f"{model_name}_best_model.pth"
            model.save_model(model_path)

            print(f"{model_name} training completed!")

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
        print("Starting full diabetic retinopathy classification pipeline...")
        print("=" * 70)

        try:
            # Setup
            self.setup_project()

            # Data preparation
            self.prepare_data()

            # Model initialization
            self.initialize_models()

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
