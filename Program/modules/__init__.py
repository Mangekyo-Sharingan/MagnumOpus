"""
Modules package for diabetic retinopathy classification
"""

__version__ = "1.0.0"
__author__ = "Adam Hagelin"

from .config import Config
from .data import CustomDataLoader as DataLoader, DiabetitcRetinopathyDataset, PipelineA, PipelineB
from .models import BaseModel, VGG16Model, ResNet50Model, InceptionV3Model, ModelFactory
from .train import Trainer, TrainingMonitor
from .model_loader import ModelLoader, load_model_for_prediction, compare_models_prediction

# Import utilities from the utils package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import Utils, Visualizer, Logger, DeviceManager
from test import Evaluator, MetricsCalculator

__all__ = [
    'Config',
    'DataLoader', 'DiabetitcRetinopathyDataset', 'PipelineA', 'PipelineB',
    'BaseModel', 'VGG16Model', 'ResNet50Model', 'InceptionV3Model', 'ModelFactory',
    'Trainer', 'TrainingMonitor',
    'Evaluator', 'MetricsCalculator',
    'Utils', 'Visualizer', 'Logger', 'DeviceManager',
    'ModelLoader', 'load_model_for_prediction', 'compare_models_prediction'
]
