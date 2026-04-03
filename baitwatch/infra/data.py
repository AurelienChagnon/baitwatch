"""Baitwatch - Data Infrastructure.

This module handles the data loading and saving.
"""

import shutil
from pathlib import Path
from typing import Literal

from google.cloud import storage
from google.cloud.storage import transfer_manager
from PIL import Image
from tensorflow import keras
from tensorflow.data import Dataset

from baitwatch.logger import logger
from baitwatch.settings import DATASET_NAME, PROJECT_PATH, cloud_settings, dataset_settings


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
        logger.info("[SUCCESS] Data already downloaded !")
        return

    logger.info(f"[DOWNLOAD] Loading data from {cloud_settings.BUCKET_NAME}...")
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
    logger.debug(f"Downloading {len(blobs)} files from bucket")
    transfer_manager.download_many_to_path(bucket,
                                           blobs,
                                           destination_directory=str(local_filename),
                                           skip_if_exists=True,
                                           )
    logger.info("[SUCCESS] Data downloaded successfully !")


def get_images(
        path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
        image_size: tuple[int, int] = dataset_settings.ORIGINAL_SIZE,
) -> tuple[Dataset, Dataset, Dataset]:
    """Retrieve images from the dataset.

    Args:
        path (Path, optional):
            Path to dataset, must contain directories 'train', 'test' and 'val'.
            Defaults to dataset_settings.RAW_DATA_PATH / DATASET_NAME.
        image_size (tuple[int, int], optional): Size to resize images to.
            Defaults to dataset_settings.ORIGINAL_SIZE.

    Returns:
        tuple[Dataset, Dataset, Dataset]:
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
    images_train = keras.utils.image_dataset_from_directory(path / "images" / "train",
                                                               labels=None,
                                                               batch_size=None,
                                                               shuffle=False,
                                                               image_size=image_size)
    images_test = keras.utils.image_dataset_from_directory(path / "images" / "test",
                                                              labels=None,
                                                              batch_size=None,
                                                              shuffle=False,
                                                              image_size=image_size)
    images_val = keras.utils.image_dataset_from_directory(path / "images" / "valid",
                                                             labels=None,
                                                             batch_size=None,
                                                             shuffle=False,
                                                             image_size=image_size)

    return images_train, images_val, images_test


def get_labels(
        path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
) -> tuple[Dataset, Dataset, Dataset]:
    """Retrieve labels from the dataset.

    Args:
        path (Path, optional):
            Path to dataset, must contain directories 'train', 'test' and 'val'.
            Defaults to dataset_settings.RAW_DATA_PATH / DATASET_NAME.

    Returns:
        tuple[Dataset, Dataset, Dataset]:
            - labels_train: training labels
            - labels_val: validation labels
            - labels_test: test labels

    Raises:
        FileNotFoundError: If no data is found at the specified path.
    """
    if not list(path.iterdir()):
        error = f"No data found at {path}"
        raise FileNotFoundError(error)

    labels_train = keras.utils.text_dataset_from_directory(path / "labels" / "train",
                                                              labels=None,
                                                              batch_size=None,
                                                              shuffle=False)
    labels_test = keras.utils.text_dataset_from_directory(path / "labels" / "test",
                                                             labels=None,
                                                             batch_size=None,
                                                             shuffle=False)
    labels_val = keras.utils.text_dataset_from_directory(path / "labels" / "valid",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False)

    return labels_train, labels_val, labels_test


def _clear_directory(path: Path) -> None:
    """Clear all contents of a directory.

    Args:
        path (Path): Directory path to clear.
    """
    logger.debug(f"Clearing directory contents: {path}")
    if not str(path.absolute()).startswith(str(PROJECT_PATH)):
        logger.warning(f"Path {path} not in project, will not clear directory.")
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def get_processed_dataset(
        path: Path = dataset_settings.PROCESSED_DATA_PATH,
        *,
        image_size: tuple[int, int],
        label_mode: Literal["int", "categorical", "auto"] = "auto",
) -> tuple[Dataset, Dataset, Dataset]:
    """Load preprocessed images into Dataset with labels.

    Args:
        path (Path): Path of preprocessed data, with train, val, test folders.
        image_size (tuple[int, int]): Size of image.
        label_mode (Literal["int", "categorical", "auto"], optional): Type of labels.
            Defaults to "auto".

    Returns:
        tuple[Dataset, Dataset, Dataset]:
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

    x_train_ds = keras.utils.image_dataset_from_directory(path / "train",
                                                             labels="inferred",
                                                             shuffle=True,
                                                             image_size=image_size,
                                                             label_mode=label_mode
                                                             )
    x_val_ds = keras.utils.image_dataset_from_directory(path / "val",
                                                           labels="inferred",
                                                           shuffle=True,
                                                           image_size=image_size,
                                                           label_mode=label_mode)
    x_test_ds = keras.utils.image_dataset_from_directory(path / "test",
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
        logger.info("[DOWNLOAD] Augmented data not found, downloading from bucket...")
        client = storage.Client()
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
        blobs = [
            blob.name
            for blob in client.list_blobs(
                cloud_settings.BUCKET_NAME,
                prefix="augmented_images",
            )
        ]
        logger.debug(f"Downloading {len(blobs)} augmented files")
        transfer_manager.download_many_to_path(
            bucket,
            blobs,
            destination_directory=str(directory_path),
            skip_if_exists=True,
        )
        logger.info("[SUCCESS] Augmented data downloaded !")
    else:
        logger.info("[SUCCESS] You already have the augmented data !")


def save_dataset_by_label(
        dataset: Dataset,
        path: Path,
        labels: Dataset | None = None,
) -> None:
    """Save a dataset with images and labels, organizing images by label.

    Images are saved into subdirectories named after their label values (0, 1, 2, etc.).
    Each image is saved as a JPEG file with a sequential index.

    Args:
        dataset (Dataset): Dataset containing (image, label) tuples or image only (use
        param labels).
        path (Path): Root path where label subdirectories will be created.
        labels (Dataset | None, optional): Dataset containing labels. Defaults to None.
    """
    if not path.exists():
        path.mkdir(parents=True)

    if list(path.iterdir()):
        logger.warning(f"Path {path} not empty, clearing contents before saving.")
        _clear_directory(path)

    label_counters = {}

    # When dataset does not contain labels, use labels param
    iterable = zip(dataset, labels, strict=True) if labels is not None else dataset.unbatch()

    for image, label in iterable:
        label_value = label.numpy().astype("int")
        label_dir = path / str(label_value)

        if not label_dir.exists():
            label_dir.mkdir(parents=True)
            label_counters[label_value] = 0

        if label_value not in label_counters:
            label_counters[label_value] = 0

        numpy_image = image.numpy().astype("uint8")
        img = Image.fromarray(numpy_image)
        img.save(label_dir / f"img_{label_counters[label_value]}.jpg")
        label_counters[label_value] += 1

    logger.info(f"Saved dataset to {path} with {len(label_counters)} label directories")
