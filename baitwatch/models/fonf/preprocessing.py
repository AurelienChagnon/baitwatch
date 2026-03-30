"""Fonf model preprocessing."""

import numpy as np
from tensorflow.data import Dataset

from baitwatch.models.preprocessing import preprocess_ds, resize_ds
from baitwatch.settings import fonf_settings


def make_training_data_fonf(
        imgs: Dataset,
        labels: Dataset,
) -> tuple[Dataset, np.ndarray]:
    """Transform dataset into training data for Fonf model.

    Args:
        imgs (Dataset): Dataset of images
        labels (Dataset): Dataset of labels

    Returns:
        tuple[Dataset, np.ndarray]: Tuple of preprocessed images and targets
    """
    x = preprocess_fonf(imgs)
    y = get_target_fonf(labels)
    return x, y


def preprocess_fonf(dataset: Dataset) -> Dataset:
    """Preprocess dataset for Fonf model.

    Args:
        dataset (Dataset): Dataset of images

    Returns:
        Dataset: Preprocessed dataset
    """
    dataset = preprocess_ds(dataset)
    dataset = resize_ds(dataset, img_size=fonf_settings.PREPROCESS_IMG_SIZE)
    return dataset


def get_target_fonf(
        labels: Dataset,
) -> np.ndarray:
    """Get the binary target "Fish Or No Fish" (fonf).

    If labels empty: no fish = O
    If labels contains something: fish = 1

    Args:
        labels (Dataset): Dataset of labels

    Returns:
        np.ndarray: Array of 0 and 1
    """
    # If there is no label, there is no fish (0)
    y = np.array([0 if txt == b'' else 1
                  for txt in labels.as_numpy_iterator()])

    return y
