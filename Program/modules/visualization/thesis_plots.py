import os
import sys
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Add parent directories to path to allow importing modules
current_dir = Path(__file__).parent
modules_dir = current_dir.parent
program_dir = modules_dir.parent
sys.path.append(str(modules_dir))
sys.path.append(str(program_dir))

try:
    from modules.config import Config
    from modules.data import CustomDataLoader
except ImportError:
    # Fallback if running from different context
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from Program.modules.config import Config
    from Program.modules.data import CustomDataLoader

def plot_dr_distribution(data_loader, output_dir):
    """
    Generates a bar chart of DR grade distribution (0-4).
    """
    print("Generating DR distribution plot...")
    
    # Ensure data is loaded
    if data_loader.merged_train_df is None:
        data_loader.load_data()
    
    df = data_loader.merged_train_df
    
    # Count diagnosis frequencies
    counts = df['diagnosis'].value_counts().sort_index()
    
    # Prepare data for plotting
    grades = [0, 1, 2, 3, 4]
    grade_labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative']
    values = [counts.get(g, 0) for g in grades]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grades, values, color=["#1f77b4", "#258cd6", "#4788b6", "#104870", "#5ba9e0"])
    
    # Add labels and title
    plt.xlabel('Diabetic Retinopathy Grade', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Distribution of Diabetic Retinopathy Grades in Training Set', fontsize=14)
    plt.xticks(grades, grade_labels, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = output_dir / 'dr_grade_distribution.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved distribution plot to {output_path}")
    plt.close()

def get_old_transforms():
    """Reconstructs the 'Old' augmentation pipeline (V1 Baseline)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Note: 'Zoom' was not explicitly in V1 code, but user mentioned it. 
        # If strictly following V1 code, it was just Resize+Flip+Rotation.
        # We will stick to the V1 methodology description.
    ])

def get_new_transforms():
    """Reconstructs the 'New' augmentation pipeline (V2 Enhanced)"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])

def visualize_augmentation_grid(image_path, transform, title, output_path):
    """
    Creates a grid with Original image on left and 3 augmented versions on right.
    """
    print(f"Generating augmentation grid: {title}...")
    
    try:
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        plt.suptitle(title, fontsize=16)
        
        # Layout: 1 row, 2 columns (Left: Original, Right: Grid of 3 augmented)
        # Actually, let's do 1 row, 4 columns: 1 Original, 3 Augmented
        
        # Subplot 1: Original
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(original_img)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Generate 3 augmented versions
        for i in range(3):
            ax = plt.subplot(1, 4, i + 2)
            
            # Apply transform
            # Note: transforms usually return Tensor if ToTensor is included.
            # Our get_old_transforms/get_new_transforms don't have ToTensor/Normalize 
            # to make visualization easier (we want PIL images back).
            
            aug_img = transform(original_img)
            
            ax.imshow(aug_img)
            ax.set_title(f"Augmentation {i+1}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved augmentation grid to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing augmentation for {image_path}: {e}")

def visualize_center_padding(image_path, target_size, output_path):
    """
    Visualizes the center padding preprocessing step.
    """
    print("Generating Center Padding visualization...")
    
    try:
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Implement center padding logic (replicated from PipelineA/B)
        # 1. Resize with aspect ratio
        original_width, original_height = original_img.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Width > Height
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:  # Height >= Width
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
            
        resized_img = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 2. Create square canvas and center
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        canvas.paste(resized_img, (x_offset, y_offset))
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image\n({original_width}x{original_height})")
        axes[0].axis('off')
        
        axes[1].imshow(canvas)
        axes[1].set_title(f"Center Padded Image\n({target_size}x{target_size})")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved center padding visualization to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing center padding: {e}")

def plot_weighted_distribution(data_loader, output_dir):
    """
    Generates a grouped bar chart comparing Original vs. Effective Weighted distribution.
    """
    print("Generating Weighted Distribution plot...")
    
    # Ensure data is loaded
    if data_loader.merged_train_df is None:
        data_loader.load_data()
    
    df = data_loader.merged_train_df
    
    # Count diagnosis frequencies
    counts = df['diagnosis'].value_counts().sort_index()
    
    # Prepare data
    grades = [0, 1, 2, 3, 4]
    grade_labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative']
    original_values = [counts.get(g, 0) for g in grades]
    
    # Weights from train.py
    weights = [0.5, 2.0, 1.0, 3.0, 4.0]
    
    # Calculate effective weighted values (Original * Weight)
    weighted_values = [v * w for v, w in zip(original_values, weights)]
    
    # Plotting
    x = np.arange(len(grades))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    # Original bars
    rects1 = plt.bar(x - width/2, original_values, width, label='Original Count', color='#1f77b4', alpha=0.7)
    
    # Weighted bars
    rects2 = plt.bar(x + width/2, weighted_values, width, label='Effective Weighted Impact', color='#d62728', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Diabetic Retinopathy Grade', fontsize=12)
    plt.ylabel('Sample Impact (Count × Weight)', fontsize=12)
    plt.title('Impact of Class Weighting on Training Distribution', fontsize=14)
    plt.xticks(x, grade_labels, rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height + 50,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    output_path = output_dir / 'weighted_distribution_simulation.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved weighted distribution plot to {output_path}")
    plt.close()

def plot_roc_curve(data_loader, output_dir):
    """
    Plots the ROC curve for the multi-class classification problem.
    """
    print("Generating ROC Curve plot...")
    
    # Ensure data is loaded
    if data_loader.merged_train_df is None:
        data_loader.load_data()
    
    df = data_loader.merged_train_df
    
    # Binarize the output
    y_true = label_binarize(df['diagnosis'], classes=[0, 1, 2, 3, 4])
    n_classes = y_true.shape[1]
    
    # Assuming 'predictions' column has the model's predicted probabilities for the positive class
    y_score = df[[f'pred_{i}' for i in range(n_classes)]].values
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Colors for each class
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'roc_curve.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved ROC curve plot to {output_path}")
    plt.close()

def plot_confusion_matrix(data_loader, output_dir):
    """
    Plots the confusion matrix for the model predictions.
    """
    print("Generating Confusion Matrix plot...")
    
    # Ensure data is loaded
    if data_loader.merged_train_df is None:
        data_loader.load_data()
    
    df = data_loader.merged_train_df
    
    # Binarize the output
    y_true = label_binarize(df['diagnosis'], classes=[0, 1, 2, 3, 4])
    y_pred = df[[f'pred_{i}' for i in range(5)]].values.argmax(axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(df['diagnosis'], y_pred)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False,
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix of Model Predictions', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.grid(False)
    
    plt.tight_layout()
    
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved confusion matrix plot to {output_path}")
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, classes, output_dir, model_name="Model"):
    """
    Generates a confusion matrix plot.
    """
    print(f"Generating Confusion Matrix for {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_confusion_matrix.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved confusion matrix to {output_path}")
    plt.close()
    
    # Normalized version
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path_norm = output_dir / f'{model_name.lower()}_confusion_matrix_normalized.png'
    plt.savefig(output_path_norm, dpi=300)
    plt.close()

def plot_roc_curves(y_true, y_probs, classes, output_dir, model_name="Model"):
    """
    Generates ROC curves for each class.
    """
    print(f"Generating ROC Curves for {model_name}...")
    
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    lw = 2
    
    colors = cycle(['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f'{model_name.lower()}_roc_curves.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved ROC curves to {output_path}")
    plt.close()

def plot_training_history(history_path, output_dir, model_name="Model"):
    """
    Generates Loss and Accuracy curves from training history.
    """
    print(f"Generating Training History plots for {model_name}...")
    
    try:
        history = np.load(history_path, allow_pickle=True).item()
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title(f'Training and Validation Loss - {model_name}', fontsize=16)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name.lower()}_loss_curve.png', dpi=300)
        plt.close()
        
        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy - {model_name}', fontsize=16)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name.lower()}_accuracy_curve.png', dpi=300)
        plt.close()
        
        print(f"Saved training history plots to {output_dir}")
        
    except Exception as e:
        print(f"Failed to plot training history: {e}")

def main():
    # Setup paths
    output_dir = Path(__file__).parent.parent.parent.parent / 'Docs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Config and DataLoader
    config = Config(use_logger=False)
    data_loader = CustomDataLoader(config, use_excel_logger=False)
    
    # 1. Generate Distribution Plot
    try:
        plot_dr_distribution(data_loader, output_dir)
        plot_weighted_distribution(data_loader, output_dir)
    except Exception as e:
        print(f"Failed to plot distribution: {e}")
    
    # 2. Generate Augmentation Visualizations
    # Get a sample image from the dataset
    if data_loader.merged_train_df is not None and not data_loader.merged_train_df.empty:
        # Try to find a moderate/severe case for better visualization
        sample_row = data_loader.merged_train_df[data_loader.merged_train_df['diagnosis'] >= 2].iloc[0]
        image_path = sample_row['image_path']
    else:
        # Fallback: try to find any image in the directories
        print("Dataset not loaded, searching for a sample image...")
        aptos_dir = config.aptos_train_images_dir
        sample_images = list(aptos_dir.glob('*.png'))
        if sample_images:
            image_path = str(sample_images[0])
        else:
            print("No images found to visualize!")
            return

    print(f"Using sample image: {image_path}")
    
    # Old Augmentations
    visualize_augmentation_grid(
        image_path, 
        get_old_transforms(), 
        "Baseline Augmentation Techniques", 
        output_dir / 'augmentation_old.png'
    )
    
    # New Augmentations
    visualize_augmentation_grid(
        image_path, 
        get_new_transforms(), 
        "Enhanced Augmentation Techniques", 
        output_dir / 'augmentation_new.png'
    )

    # Center Padding Visualization
    visualize_center_padding(
        image_path,
        256, # Target size for ResNet
        output_dir / 'preprocessing_center_padding.png'
    )

    # 3. Generate Weighted Distribution Plot
    try:
        plot_weighted_distribution(data_loader, output_dir)
    except Exception as e:
        print(f"Failed to plot weighted distribution: {e}")

    # 4. Generate Model Performance Plots (Confusion Matrix, ROC, History)
    results_dir = Path(__file__).parent.parent.parent.parent / 'results'
    models_dir = Path(__file__).parent.parent.parent.parent / 'models'
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    models_to_plot = ['vgg16', 'resnet50', 'inceptionv3']

    for model_name in models_to_plot:
        print(f"\nProcessing plots for {model_name}...")
        
        # Construct paths
        results_path = results_dir / f'{model_name}_evaluation_results.npy'
        history_path = models_dir / f'{model_name}_PRELIMINARY_RESULTS' / f'{model_name}_training_history.npy'
        
        display_name = {
            'vgg16': 'VGG16',
            'resnet50': 'ResNet50',
            'inceptionv3': 'InceptionV3'
        }.get(model_name, model_name)

        # Plot Evaluation Results (Confusion Matrix, ROC)
        if results_path.exists():
            try:
                # Read JSON content from the .npy file
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                if 'true_labels' in results and 'predictions' in results:
                    plot_confusion_matrix_custom(
                        results['true_labels'], 
                        results['predictions'], 
                        classes, 
                        output_dir, 
                        model_name=display_name
                    )
                
                if 'true_labels' in results and 'prediction_probs' in results:
                    plot_roc_curves(
                        results['true_labels'], 
                        results['prediction_probs'], 
                        classes, 
                        output_dir, 
                        model_name=display_name
                    )
            except Exception as e:
                print(f"Error processing {display_name} results: {e}")
        else:
            print(f"Results file not found: {results_path}")
            
        # Plot Training History
        if history_path.exists():
            plot_training_history(history_path, output_dir, model_name=display_name)
        else:
            print(f"History file not found: {history_path}")

if __name__ == "__main__":
    main()
