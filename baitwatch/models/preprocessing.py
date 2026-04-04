"""Baitwatch — Preprocessing pipelines."""

import cv2 as cv
import numpy as np
import tensorflow as tf


def white_balance(img: np.ndarray) -> np.ndarray:
    """Apply an automatic white balance on image.

    Input image is expected to come from tensorflow, which is in RGB.

    Args:
        img: RGB image in 8 bits numpy format

    Returns:
        White balanced image in numpy format
    """
    # Use LAB color space to avoid modifying luminance of image
    result = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    # White balance by averaging colors instead of gray method
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)  # noqa: PLR6104
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)  # noqa: PLR6104
    # Don't forget to convert back to RGB
    result = cv.cvtColor(result, cv.COLOR_LAB2RGB)
    return result


def contrast_enhance(img: np.ndarray) -> np.ndarray:
    """Apply an automatic contrast enhancement on image.

    Input image is expected to come from tensorflow, which is in RGB.

    Args:
        img: RGB image in 8 bits numpy format

    Returns:
        White balanced image in numpy format
    """
    # Use YCrCb color space to keep luminance
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    y, cr, cb = cv.split(ycrcb)

    # Contrast stretch
    y_stretched = cv.normalize(y, None, 0, 255, cv.NORM_MINMAX)

    # Histogram equalization
    y_enhanced = cv.equalizeHist(y_stretched)

    # Merge and convert back to RGB
    enhanced_ycrcb = cv.merge([y_enhanced, cr, cb])
    enhanced_image = cv.cvtColor(enhanced_ycrcb, cv.COLOR_YCrCb2RGB)
    return enhanced_image


def preprocess_ds(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Preprocess pipeline on image dataset.

    Args:
        dataset (tf.data.Dataset): Dataset of images

    Returns:
        tf.data.Dataset: Dataset of processed images
    """

    # To be applied to a tf.data.Dataset using 'map',
    # see https://www.tensorflow.org/api_docs/python/tf/py_function
    @tf.py_function(Tout=tf.uint8)  # 8bit image
    def preprocess(image: tf.Tensor) -> np.ndarray:
        # DO NOT MODIFY: Cast eager tensor into an OpenCV readable raw image
        img = image.numpy().astype("uint8")

        # White balance first to avoid degradation from previous processing
        white_balanced_img = white_balance(img)
        processed_img = contrast_enhance(white_balanced_img)

        return processed_img

    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


def resize_ds(
    dataset: tf.data.Dataset,
    img_size: tuple[int, int],
) -> tf.data.Dataset:
    """Resize images in dataset.

    Args:
        dataset (tf.data.Dataset): Dataset of images
        img_size (tuple[int, int]): Tuple of (height, width) to resize images to

    Returns:
        tf.data.Dataset: Dataset of resized images
    """

    @tf.py_function(Tout=tf.uint8)  # 8bit image
    def resize(processed_img: tf.Tensor) -> np.ndarray:
        processed_img = processed_img.numpy().astype("uint8")
        # Resize last in case it modifies too much for previous process
        resized_img = cv.resize(processed_img, img_size, interpolation=cv.INTER_LINEAR)
        return resized_img

    return dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
