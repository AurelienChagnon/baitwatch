"""Baitwatch — Model Selector.

Manage the different models for fish detection with unification of interfaces.
"""

import keras
from tensorflow.data import Dataset

from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.logger import logger
from baitwatch.models.fonf.model import build_model as fonf_model
from baitwatch.models.fonf.model import compile_model as fonf_compile_model
from baitwatch.models.fonf.model import get_optimizer as fonf_optimizer
from baitwatch.models.fonf.preprocessing import make_training_data_fonf, preprocess_fonf
from baitwatch.models.ifsp.model import build_model as ifsp_model
from baitwatch.models.ifsp.model import compile_model as ifsp_compile_model
from baitwatch.models.ifsp.model import get_optimizer as ifsp_optimizer
from baitwatch.models.ifsp.preprocessing import make_training_data_ifsp, preprocess_ifsp

__all__ = [
    "build_model",
    "get_compiled_model",
    "get_optimizer",
    "make_training_data",
    "preprocess",
]


def make_training_data(
    detection_type: FishDetectionEnum,
    images: Dataset,
    labels: Dataset,
) -> tuple[Dataset, Dataset]:
    """Call the training data pipeline for the given detection type.

    Args:
        detection_type: The type of fish detection to use.
        images: The raw images to process for the training data.
        labels: The raw labels to process for the training data.

    Returns:
        tuple[Dataset, Dataset]: The processed images and labels, ready for training.
    """
    detection_to_processed_imgs = {
        FishDetectionEnum.FONF: make_training_data_fonf,
        FishDetectionEnum.IFSP: make_training_data_ifsp,
    }

    logger.debug(f"Making training data for detection type: {detection_type}")
    return detection_to_processed_imgs[detection_type](images, labels)


def preprocess(detection_type: FishDetectionEnum, images: Dataset) -> Dataset:
    """Return the preprocessing pipeline for the given detection type.

    Args:
        detection_type: The type of fish detection to use.
        images: The raw images to process.

    Returns:
        Dataset: The preprocessed images.
    """
    detection_to_process_pipeline = {
        FishDetectionEnum.FONF: preprocess_fonf,
        FishDetectionEnum.IFSP: preprocess_ifsp,
    }

    logger.debug(f"Preprocessing images for detection type: {detection_type}")
    return detection_to_process_pipeline[detection_type](images)


def build_model(detection_type: FishDetectionEnum) -> keras.models.Model:
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
    logger.debug(f"Building model for detection type: {detection_type}")
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
    logger.debug(f"Compiling model for detection type: {detection_type}")
    model = build_model(detection_type)
    optimizer = get_optimizer(detection_type)
    compiled_model = detection_to_model_compile[detection_type](model, optimizer)
    logger.info(f"Model compiled successfully for {detection_type}")
    return compiled_model
