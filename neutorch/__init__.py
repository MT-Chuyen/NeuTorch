"""NeuTorch: lightweight, zero-boilerplate Computer Vision training on PyTorch."""

from neutorch.engine import Callback, Trainer, train_vision_task
from neutorch.metrics import (
    ClassificationEvaluator,
    DetectionEvaluator,
    SegmentationEvaluator,
    build_evaluator,
)
from neutorch.utils import get_device, to_device

__version__ = "0.1.1"

__all__ = [
    "Callback",
    "Trainer",
    "train_vision_task",
    "ClassificationEvaluator",
    "DetectionEvaluator",
    "SegmentationEvaluator",
    "build_evaluator",
    "get_device",
    "to_device",
    "__version__",
]
