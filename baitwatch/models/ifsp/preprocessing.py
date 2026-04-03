"""Preprocessing pipelines for IFSP model."""

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from baitwatch.models.ifsp.bounding_box import extract_fish_bounding_boxes, padded_resize
from baitwatch.models.preprocessing import preprocess_ds
from baitwatch.settings import ifsp_settings


def make_training_data_ifsp(
        imgs: Dataset,
        labels: Dataset,
) -> tuple[Dataset, Dataset]:
    """Preprocess images and crop every fishes out of them.

    Args:
        imgs (Dataset): Dataset of images
        labels (Dataset): Dataset of labels

    Returns:
        tuple[Dataset, Dataset]: Tuple of preprocessed images and targets
    """
    fish_bb = []
    fish_labels = []
    imgs = preprocess_ds(imgs)
    for img, label in zip(imgs, labels, strict=True):
        fishes_in_img, associated_labels = extract_fish_bounding_boxes(
            img,
            label,
            target_size=ifsp_settings.CROP_IMG_SIZE
        )
        fish_bb.extend(fishes_in_img)
        fish_labels.extend(associated_labels)

    x = Dataset.from_tensor_slices(fish_bb)
    y = Dataset.from_tensor_slices(fish_labels)

    return x, y


def preprocess_ifsp(dataset: Dataset) -> Dataset:
    """Preprocess pipeline for IFSP model.

    Automatically white balances and enhances contrast,
    then resizes and pads images to IFSP's expected size.

    Args:
        dataset (Dataset): Dataset of images

    Returns:
        Dataset: Preprocessed dataset
    """
    dataset = preprocess_ds(dataset)

    @tf.py_function(Tout=tf.uint8)  # 8bit image
    def resize(processed_img: tf.Tensor) -> np.ndarray:
        processed_img = processed_img.numpy().astype("uint8")
        resized_img = padded_resize(processed_img, target_size=ifsp_settings.CROP_IMG_SIZE)
        return resized_img
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
