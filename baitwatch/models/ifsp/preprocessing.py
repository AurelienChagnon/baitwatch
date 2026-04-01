"""Preprocessing pipelines for IFSP model."""

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from baitwatch.models.ifsp.bounding_box import build_bbox_dataframe, crop_bb, reshape_pad_crop
from baitwatch.models.preprocessing import preprocess_ds
from baitwatch.settings import dataset_settings, ifsp_settings


def make_training_data_ifsp(
        imgs: Dataset,
        labels: Dataset,
) -> tuple[Dataset, np.ndarray]:
    """Preprocess images and crop every fishes out of them.

    Args:
        imgs (Dataset): Dataset of images
        labels (Dataset): Dataset of labels

    Returns:
        tuple[Dataset, np.ndarray]: Tuple of preprocessed images and targets
    """
    # TODO: Clean up code to save images after preprocessing rather than keeping all in memory
    # Process images
    imgs_preprocessed = preprocess_ds(imgs)

    # Fetch bounding boxes
    bb_df = build_bbox_dataframe(labels, img_size=dataset_settings.ORIGINAL_SIZE)

    # Crop and pad images to keep only bounding boxes
    crop_imgs, y = crop_bb(bb_df, imgs_preprocessed)
    padded_imgs = reshape_pad_crop(crop_imgs, format_img=ifsp_settings.CROP_IMG_SIZE)
    x = Dataset.from_tensor_slices(padded_imgs)

    # Convert
    y = np.array(y)

    return x, y


def preprocess_ifsp(dataset: Dataset) -> Dataset:
    """Preprocess pipeline for IFSP model.

    Automatically white balances and enhances contrast,
    then resizes images to IFSP's expected size.

    Args:
        dataset (Dataset): Dataset of images

    Returns:
        Dataset: Preprocessed dataset
    """
    dataset = preprocess_ds(dataset)

    @tf.py_function(Tout=tf.uint8)  # 8bit image
    def resize(processed_img: tf.Tensor) -> np.ndarray:
        processed_img = processed_img.numpy().astype("uint8")
        resized_img = reshape_pad_crop([processed_img], format_img=ifsp_settings.CROP_IMG_SIZE)
        return resized_img
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
