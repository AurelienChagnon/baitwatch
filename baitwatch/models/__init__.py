from collections.abc import Callable

import keras
import numpy as np
from tensorflow.data import Dataset

from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.models.fonf.model import build_model as fonf_model
from baitwatch.models.fonf.model import compile_model as fonf_compile_model
from baitwatch.models.fonf.model import get_optimizer as fonf_optimizer
from baitwatch.models.fonf.preprocessing import preprocess_fonf, make_training_data_fonf
from baitwatch.models.ifsp.model import build_model as ifsp_model
from baitwatch.models.ifsp.model import compile_model as ifsp_compile_model
from baitwatch.models.ifsp.model import get_optimizer as ifsp_optimizer
from baitwatch.models.ifsp.preprocessing import preprocess_ifsp, process_data_ifsp

__all__ = [
    "get_build_model",
    "get_compiled_model",
    "get_optimizer",
    "get_preprocess",
    "process_data",
]


def process_data(
        detection_type: FishDetectionEnum,
) -> Callable[[Dataset, Dataset], tuple[Dataset, np.ndarray]]:
    detection_to_processed_imgs = {
        FishDetectionEnum.FONF: make_training_data_fonf,
        FishDetectionEnum.IFSP: process_data_ifsp,
    }

    return detection_to_processed_imgs[detection_type]


def get_preprocess(detection_type: FishDetectionEnum) -> Callable[[Dataset], Dataset]:
    detection_to_process_pipeline = {
        FishDetectionEnum.FONF: preprocess_fonf,
        FishDetectionEnum.IFSP: preprocess_ifsp
    }

    return detection_to_process_pipeline[detection_type]


def get_build_model(detection_type: FishDetectionEnum) -> keras.models.Model:
    detection_to_model_builder = {
        FishDetectionEnum.FONF: fonf_model,
        FishDetectionEnum.IFSP: ifsp_model,
    }
    # Only build when requested, also rebuild when requested
    return detection_to_model_builder[detection_type]()


def get_optimizer(detection_type: FishDetectionEnum) -> keras.optimizers.Optimizer:
    detection_to_optimizer = {
        FishDetectionEnum.FONF: fonf_optimizer,
        FishDetectionEnum.IFSP: ifsp_optimizer,
    }
    return detection_to_optimizer[detection_type]()


def get_compiled_model(detection_type: FishDetectionEnum) -> keras.models.Model:
    detection_to_model_compile = {
        FishDetectionEnum.FONF: fonf_compile_model,
        FishDetectionEnum.IFSP: ifsp_compile_model,
    }
    model = get_build_model(detection_type)
    optimizer = get_optimizer(detection_type)
    return detection_to_model_compile[detection_type](model, optimizer)
