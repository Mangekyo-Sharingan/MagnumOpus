"""
Example script demonstrating how to use saved models for predictions
Run this after training models to see how to load and use them
"""
from pathlib import Path
from modules import load_model_for_prediction, compare_models_prediction

def example_single_prediction():
    """Example: Load a model and make a single prediction"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*80)

    # Find the latest trained model (you can specify exact path)
    models_dir = Path(__file__).parent.parent / "models"

    # List available models
    model_packages = [d for d in models_dir.iterdir() if d.is_dir()]

    if not model_packages:
        print("❌ No trained models found. Please train models first.")
        return

    print(f"\nAvailable trained models:")
    for i, pkg in enumerate(model_packages, 1):
        print(f"  {i}. {pkg.name}")

    # Use the first available model
    model_path = model_packages[0]
    print(f"\n📦 Using model: {model_path.name}")

    # Load the model
    model = load_model_for_prediction(model_path)

    # Show model information
    model.get_model_info()

    # Find a test image
    test_images_dir = Path(__file__).parent.parent / "Data" / "Aptos" / "test_images"
    test_images = list(test_images_dir.glob("*.png"))

    if test_images:
        test_image = test_images[0]
        print(f"\n🖼️  Testing with image: {test_image.name}")

        # Make prediction with detailed output
        result = model.predict_with_details(test_image)

        # You can also access raw result
        print(f"\nRaw prediction data:")
        print(f"  Predicted class index: {result['predicted_class']}")
        print(f"  Confidence score: {result['confidence']:.4f}")
        print(f"  All probabilities: {[f'{p:.4f}' for p in result['probabilities']]}")
    else:
        print("❌ No test images found in Data/Aptos/test_images/")


def example_batch_prediction():
    """Example: Make predictions on multiple images"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)

    models_dir = Path(__file__).parent.parent / "models"
    model_packages = [d for d in models_dir.iterdir() if d.is_dir()]

    if not model_packages:
        print("❌ No trained models found.")
        return

    model_path = model_packages[0]
    print(f"\n📦 Using model: {model_path.name}")

    # Load model
    model = load_model_for_prediction(model_path)

    # Get multiple test images
    test_images_dir = Path(__file__).parent.parent / "Data" / "Aptos" / "test_images"
    test_images = list(test_images_dir.glob("*.png"))[:5]  # First 5 images

    if not test_images:
        print("❌ No test images found.")
        return

    print(f"\n🖼️  Processing {len(test_images)} images...")

    # Make batch predictions
    results = model.predict_batch(test_images)

    # Display results in a table
    print(f"\n{'Image':<30} {'Prediction':<25} {'Confidence':<12}")
    print("-"*70)

    for result in results:
        image_name = Path(result['image_path']).name
        prediction = result['class_names'][result['predicted_class']]
        confidence = result['confidence'] * 100
        print(f"{image_name:<30} {prediction:<25} {confidence:6.2f}%")


def example_compare_models():
    """Example: Compare predictions from multiple models"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Multiple Models")
    print("="*80)

    models_dir = Path(__file__).parent.parent / "models"
    model_packages = [d for d in models_dir.iterdir() if d.is_dir()]

    if len(model_packages) < 2:
        print("❌ Need at least 2 trained models for comparison.")
        print(f"   Currently have: {len(model_packages)} model(s)")
        return

    print(f"\n📦 Found {len(model_packages)} trained models")
    for pkg in model_packages:
        print(f"   • {pkg.name}")

    # Get a test image
    test_images_dir = Path(__file__).parent.parent / "Data" / "Aptos" / "test_images"
    test_images = list(test_images_dir.glob("*.png"))

    if not test_images:
        print("❌ No test images found.")
        return

    test_image = test_images[0]

    # Compare models
    results = compare_models_prediction(model_packages, test_image)

    # Analyze consensus
    predictions = [r['predicted_class'] for r in results]
    if len(set(predictions)) == 1:
        print(f"\n✓ All models agree on the prediction!")
    else:
        print(f"\n⚠️  Models have different predictions - review recommended")


def example_save_predictions_to_csv():
    """Example: Process images and save results to CSV"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Save Predictions to CSV")
    print("="*80)

    import pandas as pd

    models_dir = Path(__file__).parent.parent / "models"
    model_packages = [d for d in models_dir.iterdir() if d.is_dir()]

    if not model_packages:
        print("❌ No trained models found.")
        return

    model_path = model_packages[0]
    print(f"\n📦 Using model: {model_path.name}")

    # Load model
    model = load_model_for_prediction(model_path)

    # Get test images
    test_images_dir = Path(__file__).parent.parent / "Data" / "Aptos" / "test_images"
    test_images = list(test_images_dir.glob("*.png"))[:10]  # First 10

    if not test_images:
        print("❌ No test images found.")
        return

    print(f"\n🖼️  Processing {len(test_images)} images...")

    # Make predictions
    results = model.predict_batch(test_images)

    # Convert to DataFrame
    data = []
    for result in results:
        data.append({
            'image_name': Path(result['image_path']).name,
            'predicted_class': result['predicted_class'],
            'prediction_label': result['class_names'][result['predicted_class']],
            'confidence': result['confidence'],
            'prob_no_dr': result['probabilities'][0],
            'prob_mild': result['probabilities'][1],
            'prob_moderate': result['probabilities'][2],
            'prob_severe': result['probabilities'][3],
            'prob_proliferative': result['probabilities'][4],
        })

    df = pd.DataFrame(data)

    # Save to CSV
    output_path = Path(__file__).parent.parent / "results" / "predictions_example.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"\nFirst few predictions:")
    print(df[['image_name', 'prediction_label', 'confidence']].head())


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("TRAINED MODEL USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates how to:")
    print("  1. Load a trained model")
    print("  2. Make predictions on single/multiple images")
    print("  3. Compare predictions from multiple models")
    print("  4. Save predictions to CSV")
    print("\nMake sure you have trained at least one model first!")

    try:
        example_single_prediction()
        example_batch_prediction()
        example_compare_models()
        example_save_predictions_to_csv()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*80)
        print("\nNext steps:")
        print("  • Check MODEL_USAGE_GUIDE.md for detailed documentation")
        print("  • Modify these examples for your specific use case")
        print("  • Integrate model loading into your application")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

