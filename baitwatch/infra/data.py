"""Baitwatch - Data Infrastructure.

This module handles the data loading and saving.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from google.cloud import storage
from google.cloud.storage import transfer_manager
from PIL import Image

from baitwatch.settings import DATASET_NAME, cloud_settings, dataset_settings


def dl_data(
        path: Path = dataset_settings.RAW_DATA_PATH
) -> None:
    """Download Baitwatch dataset from Cloud Storage.

    Args:
        path (Path, optional): Path to save the dataset.
            Defaults to dataset_settings.RAW_DATA_PATH.
    """
    datadir_path = path / DATASET_NAME

    if datadir_path.is_dir() and list(datadir_path.iterdir()):
        print("✅ Data already downloaded !")

    else:
        print(f"✋ Loading data from {cloud_settings.BUCKET_NAME}...")
        local_filename = path

        client = storage.Client()
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
        blobs = [
            blob.name
            for blob in client.list_blobs(
                cloud_settings.BUCKET_NAME,
                prefix="training_data_species_grouped",
            )
        ]
        transfer_manager.download_many_to_path(bucket,
                                               blobs,
                                               destination_directory=str(local_filename),
                                               skip_if_exists=True,
                                               )
        print("✅ Data downloaded successfully !")


def get_images(
        path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
        image_size: tuple[int, int] = dataset_settings.ORIGINAL_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Retrieve images from the dataset.

    Args:
        path (Path, optional):
            Path to dataset, must contain directories 'train', 'test' and 'val'.
            Defaults to dataset_settings.RAW_DATA_PATH / DATASET_NAME.
        image_size (tuple[int, int], optional): Size to resize images to.
            Defaults to dataset_settings.ORIGINAL_SIZE.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            - images_train: training images
            - images_val: validation images
            - images_test: test images

    Raises:
        FileNotFoundError: If no data is found at the specified path.
    """
    if not list(path.iterdir()):
        error = f"No data found at {path}"
        raise FileNotFoundError(error)

    # image_dataset_from_directory retrieves images from the directory
    images_train = tf.keras.utils.image_dataset_from_directory(path / "images" / "train",
                                                               labels=None,
                                                               batch_size=None,
                                                               shuffle=False,
                                                               image_size=image_size)
    images_test = tf.keras.utils.image_dataset_from_directory(path / "images" / "test",
                                                              labels=None,
                                                              batch_size=None,
                                                              shuffle=False,
                                                              image_size=image_size)
    images_val = tf.keras.utils.image_dataset_from_directory(path / "images" / "valid",
                                                             labels=None,
                                                             batch_size=None,
                                                             shuffle=False,
                                                             image_size=image_size)

    return images_train, images_val, images_test


def get_labels(
        path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Retrieve labels from the dataset.

    Args:
        path (Path, optional):
            Path to dataset, must contain directories 'train', 'test' and 'val'.
            Defaults to dataset_settings.RAW_DATA_PATH / DATASET_NAME.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            - labels_train: training labels
            - labels_val: validation labels
            - labels_test: test labels

    Raises:
        FileNotFoundError: If no data is found at the specified path.
    """
    if not list(path.iterdir()):
        error = f"No data found at {path}"
        raise FileNotFoundError(error)

    labels_train = tf.keras.utils.text_dataset_from_directory(path / "labels" / "train",
                                                              labels=None,
                                                              batch_size=None,
                                                              shuffle=False)
    labels_test = tf.keras.utils.text_dataset_from_directory(path / "labels" / "test",
                                                             labels=None,
                                                             batch_size=None,
                                                             shuffle=False)
    labels_val = tf.keras.utils.text_dataset_from_directory(path / "labels" / "valid",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False)

    return labels_train, labels_val, labels_test


def save_image_dataset(
        dataset: tf.data.Dataset,
        path: Path,
        labels: np.ndarray | None = None,
) -> None:
    """Save the dataset as JPEG images.

    If labels is passed, the images are separated into different folder according
    to the labels.
    Labels MUST BE ordered accordingly to associate correctly the image in dataset.

    Args:
        dataset (tf.data.Dataset): Dataset to save.
        path (Path): Path to save dataset into.
        labels (np.ndarray | None, optional): Labels to separate dataset into. Defaults to None.
    """
    if not path.exists():
        path.mkdir(parents=True)

    if list(path.iterdir()):
        print(f"Warning! Path {path} not empty, images will be rewritten.")

    # Dataset are not loaded files, len(dataset) would only return 1
    len_dataset = dataset.cardinality().numpy()

    if labels is None:
        # Create an array of empty strings so no label directories are needed
        labels = np.array(["" for _ in range(len_dataset)])
    else:
        # Create directories for each label
        for label in np.unique(labels):
            label_path = path / str(label)
            if not label_path.exists():
                label_path.mkdir(parents=True)

    for index, (tensor, label) in enumerate(zip(dataset, labels, strict=True)):
        # Cast into numpay array
        numpy_image = tensor.numpy().astype("uint8")
        image = Image.fromarray(numpy_image)
        image.save(path / str(label) / f"img_{index}.jpg")


def get_processed_dataset(
        path: Path = dataset_settings.PROCESSED_DATA_PATH,
        *,
        image_size: tuple[int, int],
        label_mode: Literal["int", "categorical", "auto"] = "auto",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load preprocessed images into tf.data.Dataset with labels.

    Args:
        path (Path): Path of preprocessed data, with train, val, test folders.
        image_size (tuple[int, int]): Size of image.
        label_mode (Literal["int", "categorical", "auto"], optional): Type of labels.
            Defaults to "auto".

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            - x_train_ds: training dataset
            - x_val_ds: validation dataset
            - x_test_ds: test dataset

    Raises:
        FileNotFoundError: If no data is found at the specified path.
    """
    if not list(path.iterdir()):
        error = f"No data found at {path}"
        raise FileNotFoundError(error)

    if label_mode == "auto":
        # Check one directory to get label mode: int for bi-class, categorical for multi class
        test_path = path / "train"
        categorical_threshold = 2
        nb_dir = len(
            [f for f in test_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
        )  # Ignore hidden directories
        label_mode = "categorical" if nb_dir > categorical_threshold else "int"

    x_train_ds = tf.keras.utils.image_dataset_from_directory(path / "train",
                                                             labels="inferred",
                                                             shuffle=True,
                                                             image_size=image_size,
                                                             label_mode=label_mode
                                                             )
    x_val_ds = tf.keras.utils.image_dataset_from_directory(path / "val",
                                                           labels="inferred",
                                                           shuffle=True,
                                                           image_size=image_size,
                                                           label_mode=label_mode)
    x_test_ds = tf.keras.utils.image_dataset_from_directory(path / "test",
                                                            labels="inferred",
                                                            shuffle=True,
                                                            image_size=image_size,
                                                            label_mode=label_mode)

    return x_train_ds, x_val_ds, x_test_ds


def dl_augmented_images(
    directory_path: Path = dataset_settings.RAW_DATA_PATH,
    ) -> None:
    """Loads augmented images from local storage.

    If not available, downloads them from the bucket first.

    Args:
        directory_path (Path): local path to raw_data/
    """
    augmented_path = directory_path / 'augmented_images'

    # ── Download if not available locally ────────────────────────
    if (not augmented_path.exists()
            or not [f for f in augmented_path.iterdir() if not f.name.startswith('.')]):
        print("✋ Augmented data not found, downloading from bucket...")
        client = storage.Client()
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
        blobs = [
            blob.name
            for blob in client.list_blobs(
                cloud_settings.BUCKET_NAME,
                prefix="augmented_images",
            )
        ]
        transfer_manager.download_many_to_path(
            bucket,
            blobs,
            destination_directory=str(directory_path),
            skip_if_exists=True,
        )
        print("✅ Augmented data downloaded !")
    else:
        print("✅ You already have the augmented data !")


def save_augmented_to_local(dataset: tf.data.Dataset, model_name: str, split: str) -> None:
    """Applies data augmentation to a dataset and saves the results to local storage.

    This function processes an input dataset using a flat_map transformation to
    generate multiple augmented variations (images and labels) for every original
    sample. It then collects these variations into memory and exports them as
    individual files using the project's standardized saving utility.

    Args:
        dataset (tf.data.Dataset): The input dataset containing (image, label) pairs.
            It is recommended to unbatch the dataset before passing it to this function.
        model_name (str): The name of the model/species task (e.g., 'ifsp', 'fonf'),
            used to define the output directory.
        split (str): The dataset split being processed (e.g., 'train', 'val', or 'test').
    """
    images = []
    labels = []
    for img, lab in dataset:
        images.append(img)
        labels.append(lab)

    images = tf.concat(images, axis=0)
    labels = tf.concat(labels, axis=0)

    save_image_dataset(tf.data.Dataset.from_tensor_slices(images),
                       dataset_settings.PROCESSED_DATA_PATH / f'{model_name}_augmented' / split,
                       labels=labels.numpy())
