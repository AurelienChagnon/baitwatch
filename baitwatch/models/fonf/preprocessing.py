import numpy as np
from tensorflow.data import Dataset

from baitwatch.models.commons.preprocessing import preprocess_ds, resize_ds
from baitwatch.settings import fonf_settings


def process_data_fonf(
        imgs: Dataset,
        labels: Dataset,
) -> tuple[Dataset, np.ndarray]:
    x = preprocess_fonf(imgs)
    y = get_target_fonf(labels)
    return x, y


def preprocess_fonf(dataset: Dataset) -> Dataset:
    dataset = preprocess_ds(dataset)
    dataset = resize_ds(dataset, img_size=fonf_settings.PREPROCESS_IMG_SIZE)
    return dataset


def get_target_fonf(
        labels: Dataset,
) -> np.ndarray:
    """Get the binary target "Fish Or No Fish" (fonf)
    If no labels : no fish = O
    If labels : fish = 1

    Returns:
        the targets for train, val and test (arrays of 0 and 1)
    """
    # If there is no label, there is no fish (0)
    y = np.array([0 if txt == b'' else 1
                  for txt in labels.as_numpy_iterator()])

    return y
