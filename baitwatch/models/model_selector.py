"""Baitwatch — Model Selector.

Manage the different models for fish detection with unification of interfaces.
"""

from collections.abc import Callable

import keras
import numpy as np
from tensorflow.data import Dataset

from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.models.fonf.model import build_model as fonf_model
from baitwatch.models.fonf.model import compile_model as fonf_compile_model
from baitwatch.models.fonf.model import get_optimizer as fonf_optimizer
from baitwatch.models.fonf.preprocessing import make_training_data_fonf, preprocess_fonf
from baitwatch.models.ifsp.model import build_model as ifsp_model
from baitwatch.models.ifsp.model import compile_model as ifsp_compile_model
from baitwatch.models.ifsp.model import get_optimizer as ifsp_optimizer
from baitwatch.models.ifsp.preprocessing import make_training_data_ifsp, preprocess_ifsp

__all__ = [
    "get_build_model",
    "get_compiled_model",
    "get_optimizer",
    "get_preprocess",
    "make_training_data",
]


def make_training_data(
        detection_type: FishDetectionEnum,
) -> Callable[[Dataset, Dataset], tuple[Dataset, np.ndarray]]:
    """Return the training data pipeline for the given detection type.

    Args:
        detection_type: The type of fish detection to use.

    Returns:
        A callable that takes two datasets (images and labels) and returns a tuple of datasets.
    """
    detection_to_processed_imgs = {
        FishDetectionEnum.FONF: make_training_data_fonf,
        FishDetectionEnum.IFSP: make_training_data_ifsp,
    }

    return detection_to_processed_imgs[detection_type]


def get_preprocess(detection_type: FishDetectionEnum) -> Callable[[Dataset], Dataset]:
    """Return the preprocessing pipeline for the given detection type.

    Args:
        detection_type: The type of fish detection to use.

    Returns:
        A callable that takes a dataset and returns a preprocessed dataset.
    """
    detection_to_process_pipeline = {
        FishDetectionEnum.FONF: preprocess_fonf,
        FishDetectionEnum.IFSP: preprocess_ifsp
    }

    return detection_to_process_pipeline[detection_type]


def get_build_model(detection_type: FishDetectionEnum) -> keras.models.Model:
    """Return the model for the given detection type.

    Args:
        detection_type: The type of fish detection to use.

    Returns:
        The model for the given detection type.
    """
    detection_to_model_builder = {
        FishDetectionEnum.FONF: fonf_model,
        FishDetectionEnum.IFSP: ifsp_model,
    }
    # Only build when requested, also rebuild when requested
    return detection_to_model_builder[detection_type]()


def get_optimizer(detection_type: FishDetectionEnum) -> keras.optimizers.Optimizer:
    """Return the optimizer for the given detection type.

    Args:
        detection_type: The type of fish detection to use.

    Returns:
        The optimizer for the given detection type.
    """
    detection_to_optimizer = {
        FishDetectionEnum.FONF: fonf_optimizer,
        FishDetectionEnum.IFSP: ifsp_optimizer,
    }
    return detection_to_optimizer[detection_type]()


def get_compiled_model(detection_type: FishDetectionEnum) -> keras.models.Model:
    """Return the compiled model for training, given the detection type.

    Args:
        detection_type: The type of fish detection to use.

    Returns:
        The compiled model for the given detection type.
    """
    detection_to_model_compile = {
        FishDetectionEnum.FONF: fonf_compile_model,
        FishDetectionEnum.IFSP: ifsp_compile_model,
    }
    model = get_build_model(detection_type)
    optimizer = get_optimizer(detection_type)
    return detection_to_model_compile[detection_type](model, optimizer)
