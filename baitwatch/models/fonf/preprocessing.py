"""Fonf model preprocessing."""

from tensorflow.data import Dataset

from baitwatch.logger import logger
from baitwatch.models.preprocessing import preprocess_ds, resize_ds
from baitwatch.settings import fonf_settings


def make_training_data_fonf(
    imgs: Dataset,
    labels: Dataset,
) -> tuple[Dataset, Dataset]:
    """Transform dataset into training data for Fonf model.

    Args:
        imgs (Dataset): Dataset of images
        labels (Dataset): Dataset of labels

    Returns:
        tuple[Dataset, Dataset]: Tuple of preprocessed images and targets
    """
    logger.debug("Starting FONF training data preparation")
    x = preprocess_fonf(imgs)
    y = to_binary_fonf(labels)
    logger.info("FONF training data prepared successfully")
    return x, y


def preprocess_fonf(dataset: Dataset) -> Dataset:
    """Preprocess dataset for Fonf model.

    Automatically white balances and enhances contrast,
    then resizes images to Fonf's expected size (see settings).

    Args:
        dataset (Dataset): Dataset of images

    Returns:
        Dataset: Preprocessed dataset
    """
    logger.debug("Preprocessing dataset for FONF model")
    dataset = preprocess_ds(dataset)
    dataset = resize_ds(dataset, img_size=fonf_settings.PREPROCESS_IMG_SIZE)
    return dataset


def to_binary_fonf(y: Dataset) -> Dataset:
    """Convert labels to binary for Fonf model.

    Fish = 1
    No fish = 0

    Args:
        y (Dataset): Dataset of labels

    Returns:
        Dataset: Dataset of binary labels
    """
    return y.map(lambda x: 0 if x == b"" else 1)
