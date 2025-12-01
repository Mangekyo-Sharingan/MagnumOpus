#!/usr/bin/env python3
"""
Test script to verify BokehLivePlotter is working
"""

import sys
from pathlib import Path

# Add Program directory to path
sys.path.insert(0, str(Path(__file__).parent / "Program"))

from modules.train import BokehLivePlotter

def test_bokeh_plotter():
    """Test the BokehLivePlotter class"""
    print("=" * 70)
    print("Testing BokehLivePlotter")
    print("=" * 70)

    try:
        # Create plotter
        output_dir = Path(__file__).parent / "logs" / "live_plots"
        print(f"\n1. Creating BokehLivePlotter...")
        print(f"   Output directory: {output_dir}")

        plotter = BokehLivePlotter(output_dir, model_name='test_model')
        print(f"   [OK] Plotter created successfully")
        print(f"   HTML file will be: {plotter.html_file}")

        # Simulate some epochs
        print(f"\n2. Simulating 5 epochs of training...")
        for epoch in range(1, 6):
            train_loss = 2.5 - (epoch * 0.3)
            val_loss = 2.6 - (epoch * 0.28)
            train_acc = 0.3 + (epoch * 0.12)
            val_acc = 0.28 + (epoch * 0.11)

            print(f"   Epoch {epoch}: Loss={train_loss:.3f}, Val={val_loss:.3f}, "
                  f"Acc={train_acc:.3f}, ValAcc={val_acc:.3f}")

            plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)

        print(f"\n3. Checking if file was created...")
        if plotter.html_file.exists():
            print(f"   [OK] SUCCESS: HTML file created!")
            print(f"   File location: {plotter.html_file}")
            print(f"   File size: {plotter.html_file.stat().st_size} bytes")
            print(f"\n4. Open this file in your browser:")
            print(f"   file://{plotter.html_file.absolute()}")
        else:
            print(f"   [FAIL] FAILED: HTML file was not created")
            print(f"   Expected location: {plotter.html_file}")

        print("\n" + "=" * 70)
        print("Test complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_bokeh_plotter()

