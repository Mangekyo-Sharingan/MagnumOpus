"""
Testing and evaluation module for diabetic retinopathy classification models
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Evaluator:
    """Class responsible for evaluating trained models"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.predictions = None
        self.true_labels = None
        self.prediction_probs = None

        self.use_tta = getattr(config, 'use_tta', False)
        self.tta_flip = getattr(config, 'tta_flip', True)

    def _tta_augmentations(self, x):
        """Generate TTA augmented versions; returns list of tensors including original."""
        variants = [x]
        if self.tta_flip:
            variants.append(torch.flip(x, dims=[-1]))  # horizontal
            variants.append(torch.flip(x, dims=[-2]))  # vertical
        return variants

    def _tta_aggregate(self, logits_list):
        """Average logits from multiple TTA variants."""
        return torch.stack(logits_list, dim=0).mean(dim=0)

    def evaluate_model(self, test_loader):
        """Evaluate model performance on test data with optional TTA"""
        print("=" * 50)
        print(f"EVALUATING {self.model.model_name.upper()} MODEL")
        print("=" * 50)

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                if self.use_tta:
                    # Build variants and aggregate logits
                    logits_list = []
                    for variant in self._tta_augmentations(data):
                        outputs = self.model(variant)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        logits_list.append(outputs)
                    outputs = self._tta_aggregate(logits_list)
                else:
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                # Get probabilities and predictions
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                if batch_idx % 10 == 0:
                    print(f'Processed batch {batch_idx}/{len(test_loader)}')

        # Store results
        self.predictions = np.array(all_predictions)
        self.true_labels = np.array(all_labels)
        self.prediction_probs = np.array(all_probs)

        print(f"[OK] Evaluation completed on {len(self.predictions)} samples")
        return self.predictions, self.true_labels, self.prediction_probs

    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        print("=" * 50)
        print("CALCULATING METRICS")
        print("=" * 50)

        # Basic accuracy
        accuracy = accuracy_score(self.true_labels, self.predictions)
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None
        )

        # Weighted averages
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)

        print(f"Weighted Precision: {precision_weighted:.4f}")
        print(f"Weighted Recall: {recall_weighted:.4f}")
        print(f"Weighted F1-Score: {f1_weighted:.4f}")

        # Detailed classification report
        print("\nDetailed Classification Report:")
        class_names = [f'Class {i}' for i in range(self.config.num_classes)]
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=class_names,
            digits=4
        )
        print(report)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        return metrics

    def plot_confusion_matrix(self, save_path=None, normalize=False):
        """Plot confusion matrix"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        cm = confusion_matrix(self.true_labels, self.predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=[f'Class {i}' for i in range(self.config.num_classes)],
                   yticklabels=[f'Class {i}' for i in range(self.config.num_classes)])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_class_distribution(self, save_path=None):
        """Plot distribution of true labels vs predictions"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # True labels distribution
        unique_true, counts_true = np.unique(self.true_labels, return_counts=True)
        ax1.bar(unique_true, counts_true, alpha=0.7, color='blue')
        ax1.set_title('True Labels Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(self.config.num_classes))

        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(self.predictions, return_counts=True)
        ax2.bar(unique_pred, counts_pred, alpha=0.7, color='red')
        ax2.set_title('Predicted Labels Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(self.config.num_classes))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Class distribution plot saved to: {save_path}")

        plt.show()

    def get_misclassified_samples(self, top_k=10):
        """Get top-k misclassified samples with highest confidence"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        # Find misclassified samples
        misclassified_mask = self.predictions != self.true_labels
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("No misclassified samples found!")
            return []

        # Get confidence scores for misclassified samples
        misclassified_probs = self.prediction_probs[misclassified_indices]
        misclassified_confidence = np.max(misclassified_probs, axis=1)

        # Sort by confidence (highest first)
        sorted_indices = np.argsort(misclassified_confidence)[::-1]
        top_misclassified = misclassified_indices[sorted_indices[:top_k]]

        results = []
        for idx in top_misclassified:
            results.append({
                'index': idx,
                'true_label': self.true_labels[idx],
                'predicted_label': self.predictions[idx],
                'confidence': misclassified_confidence[sorted_indices == np.where(misclassified_indices == idx)[0][0]][0],
                'probabilities': self.prediction_probs[idx]
            })

        print(f"Top {len(results)} misclassified samples (by confidence):")
        for i, result in enumerate(results):
            print(f"{i+1}. Index: {result['index']}, True: {result['true_label']}, "
                  f"Pred: {result['predicted_label']}, Conf: {result['confidence']:.4f}")

        return results

    def save_results(self, filepath, metrics=None):
        """Save evaluation results to file"""
        results = {
            'model_name': self.model.model_name,
            'predictions': self.predictions.tolist() if self.predictions is not None else None,
            'true_labels': self.true_labels.tolist() if self.true_labels is not None else None,
            'prediction_probs': self.prediction_probs.tolist() if self.prediction_probs is not None else None,
            'metrics': metrics
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")

    def generate_classification_report(self, save_path=None):
        """Generate and optionally save a detailed classification report"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        print("\n" + "=" * 80)
        print(" " * 25 + "CLASSIFICATION REPORT")
        print("=" * 80)

        # Use config class names if available, otherwise use generic names
        if hasattr(self.config, 'class_names') and self.config.class_names:
            class_names = [self.config.class_names[i] for i in range(self.config.num_classes)]
        else:
            class_names = [f'Class {i}' for i in range(self.config.num_classes)]

        # Generate classification report
        report = classification_report(
            self.true_labels,
            self.predictions,
            target_names=class_names,
            digits=4
        )

        print(report)
        print("=" * 80 + "\n")

        # Save to file if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write("CLASSIFICATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)
                f.write("\n" + "=" * 80 + "\n")
            print(f"[OK] Classification report saved to: {save_path}")

        return report

    def generate_report(self, save_dir=None):
        """Generate comprehensive evaluation report"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("No predictions available. Run evaluate_model() first.")

        print("=" * 50)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Plot confusion matrix
        if save_dir:
            cm_path = Path(save_dir) / f"{self.model.model_name}_confusion_matrix.png"
            self.plot_confusion_matrix(save_path=cm_path)

            # Plot class distribution
            dist_path = Path(save_dir) / f"{self.model.model_name}_class_distribution.png"
            self.plot_class_distribution(save_path=dist_path)

            # Save results
            results_path = Path(save_dir) / f"{self.model.model_name}_results.json"
            self.save_results(results_path, metrics)
        else:
            self.plot_confusion_matrix()
            self.plot_class_distribution()

        # Get misclassified samples
        misclassified = self.get_misclassified_samples(top_k=5)

        print("[OK] Comprehensive report generated")
        return metrics, misclassified

# Test function for independent execution
def test_test_module():
    """Test the test/evaluation module independently"""
    print("Testing Test/Evaluation module...")

    try:
        # Import required modules
        from config import Config
        from models import get_model  # get_model exists in models.py

        # Create config
        config = Config()
        print("[OK] Config imported and created successfully")

        # Test evaluator creation for each model
        model_names = ['vgg16', 'resnet50', 'inceptionv3']

        for model_name in model_names:
            try:
                print(f"Testing Evaluator with {model_name.upper()} model...")

                # Create model
                model = get_model(model_name, config)
                print(f"[OK] {model_name.upper()} model created")

                # Create evaluator
                evaluator = Evaluator(model, config)
                print(f"[OK] Evaluator created for {model_name.upper()}")

                # Test that evaluator has all required attributes
                assert hasattr(evaluator, 'model'), "Evaluator missing model"
                assert hasattr(evaluator, 'config'), "Evaluator missing config"
                assert hasattr(evaluator, 'device'), "Evaluator missing device"

                print(f"[OK] All evaluator attributes present for {model_name.upper()}")
                print(f"[OK] {model_name.upper()} evaluator test completed")

            except Exception as e:
                print(f"[FAIL] {model_name.upper()} evaluator test failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        print("[OK] All evaluator tests successful")
        print("[OK] Test/Evaluation module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Test/Evaluation module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


class MetricsCalculator:
    """Helper class for calculating various evaluation metrics"""

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision_recall_f1(y_true, y_pred, average='weighted'):
        """Calculate precision, recall, and F1 score"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        return precision, recall, f1

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred):
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(y_true, y_pred, target_names=None):
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


if __name__ == "__main__":
    success = test_test_module()
    print(f"\nTest/Evaluation Module Test Result: {'PASS' if success else 'FAIL'}")
