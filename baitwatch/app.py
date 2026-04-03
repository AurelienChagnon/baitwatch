"""Baitwatch — Main Pipeline."""

import numpy as np
from PIL import ImageFile
from tensorflow.data import Dataset
from tensorflow.keras import Model

from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.infra.data import (
    dl_data,
    get_images,
    get_labels,
    get_processed_dataset,
    save_dataset_by_label,
)
from baitwatch.infra.registry import load_model, save_model
from baitwatch.logger import logger
from baitwatch.models.augment import augment_ds
from baitwatch.models.model_selector import get_compiled_model, make_training_data, preprocess
from baitwatch.models.training import (
    get_class_weights,
    get_classification_report,
    plot_history,
    train_model,
)
from baitwatch.settings import (
    DATASET_NAME,
    dataset_settings,
    fonf_settings,
    ifsp_settings,
    model_settings,
)

__all__ = [
    "augment",
    "classification_report",
    "download_data",
    "evaluate",
    "preprocess_data",
    "run_cycle",
    "train",
]

# Define image sizes
# REMEMBER Preprocess with Opencv
# which reverse order of image size compared to tensorflow used to load data
DETECTION_TYPE_TO_IMG_SIZE = {
    FishDetectionEnum.FONF: fonf_settings.PREPROCESS_IMG_SIZE[::-1],
    FishDetectionEnum.IFSP: ifsp_settings.CROP_IMG_SIZE,
}


def download_data() -> None:
    """Download data locally."""
    logger.info("[DOWNLOAD] Downloading data...")
    dl_data(path=dataset_settings.RAW_DATA_PATH)
    logger.info("[SUCCESS] Data downloaded")


def preprocess_data(task_type: FishDetectionEnum) -> None:
    """Process the data locally and save them for training purposes."""
    logger.info("[PREPROCESS] Starting dataset preprocessing...")
    task_type = FishDetectionEnum(task_type)
    imgs_train, imgs_val, imgs_test = get_images(
        path=dataset_settings.RAW_DATA_PATH / DATASET_NAME,
        image_size=dataset_settings.ORIGINAL_SIZE,
    )
    labels_train, labels_val, labels_test = get_labels(
        path=dataset_settings.RAW_DATA_PATH / DATASET_NAME
    )

    logger.info("   Preprocessing images...")
    x_train, y_train = make_training_data(task_type, imgs_train, labels_train)
    x_val, y_val = make_training_data(task_type, imgs_val, labels_val)
    x_test, y_test = make_training_data(task_type, imgs_test, labels_test)

    logger.info("[SAVE] Saving preprocessed datasets...")
    task_path = dataset_settings.PROCESSED_DATA_PATH / task_type.value
    save_dataset_by_label(x_train, task_path / "train", labels=y_train)
    save_dataset_by_label(x_val, task_path / "val", labels=y_val)
    save_dataset_by_label(x_test, task_path / "test", labels=y_test)

    logger.info("[SUCCESS] Preprocessing completed and saved")


def train(model_type: FishDetectionEnum, augmented: bool = False) -> None:
    """Builds, compiles and trains the model, then saves + displays the curves."""
    logger.info(f"[TRAIN] Train model({model_type})...")
    # Cast str as Enum object (from Make)
    model_type = FishDetectionEnum(model_type)

    model_dir_path = f"{model_type.value}_augmented" if augmented else f"{model_type.value}"
    x_train_ds, x_val_ds, _ = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_dir_path,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type]
    )

    logger.info(f"[BUILD] Building model {model_type}...")
    model = get_compiled_model(model_type)

    logger.info("[TRAIN] Training model...")
    class_weights = None
    if model_type == FishDetectionEnum.IFSP:
        # Manage class unbalanced
        class_weights = get_class_weights(x_train_ds, encoded=True)
        logger.debug(f"Class weights computed: {class_weights}")
    history, model = train_model(
        model,
        x_train_ds,
        validation_data=x_val_ds,
        class_weights=class_weights,
    )

    logger.info("[SAVE] Saving model...")
    save_model(model, model_type, model_settings.MODEL_LOCAL_PATH)
    logger.info("[SUCCESS] Training finished")
    plot_history(history)


def evaluate(model_type: FishDetectionEnum) -> None:
    """Evaluate the model on the test set and display the metrics."""
    logger.info(f"[EVALUATE] Model evaluating ({model_type})...")

    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH)

    _, _, x_test_ds = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type]
    )

    results = model.evaluate(x_test_ds, return_dict=True)
    logger.info(f"Evaluation results: {results}")
    logger.info("[SUCCESS] Evaluation completed")


def classification_report(model_type: FishDetectionEnum, model_name: str = "") -> None:
    """Load the model and display the classification report on the validation set."""
    logger.info(f"[REPORT] Generating classification report ({model_type})...")
    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH, model_name=model_name)

    _, x_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type.value,
                                           image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type])

    logger.info(f"\n{get_classification_report(model, x_val_ds)}")

    logger.info("[SUCCESS] Report generated")


def run_cycle(task_type: FishDetectionEnum) -> None:
    """Run the full cycle: download → preprocess → train → classification report."""
    logger.info("[CYCLE] Starting full cycle...")
    # Cast str as Enum object
    task_type = FishDetectionEnum(task_type)

    download_data()
    preprocess_data(task_type)
    train(task_type)
    classification_report(task_type)

    logger.info("[SUCCESS] Full cycle completed")


def detect_fishes(
        model: Model,
        detection_type: FishDetectionEnum,
        image: ImageFile.ImageFile,
) -> list[list[float]]:
    """Request a fish detection on given image, based on given model.

    Perform preprocessing on image then predict on processed image.

    Args:
        model (FishDetectionEnum): Fish detection model
        detection_type (FishDetectionEnum): Fish detection type
        image (ImageFile.ImageFile): Image file object

    Returns:
        List with probabilities of fish detection
    """
    # Perform preprocessing
    logger.debug(f"Preprocessing image for detection type: {detection_type}")
    image_ds = Dataset.from_tensors(np.array(image))
    image_preprocessed = preprocess(detection_type, image_ds)
    # Perform detection
    # DO NOT MODIFY, model expects a batch size
    logger.debug("Running model prediction")
    results = model.predict(image_preprocessed.batch(1))
    logger.debug(f"Detection results: {results}")
    return results


def augment(
        detection_type: FishDetectionEnum,
) -> None:
    """Orchestrates the augmentation and local storage of the given dataset splits.

    This function performs the following steps:
    1. Loads the preprocessed datasets (train, validation, and test) from
       the local processed data path using specific crop dimensions.
    2. Sequentially triggers the augmentation and saving process for each split
       ('train', 'val', 'test').

    The resulting augmented images and labels are stored in subdirectories
    corresponding to their respective model types and splits.

    Args:
        detection_type (FishDetectionEnum): Fish detection type
    """
    # Cast str as Enum object
    task_type = FishDetectionEnum(detection_type)
    logger.info(f"Loading {task_type} datasets for augmentation...")
    x_train, x_val, x_test = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / task_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[task_type],
        label_mode="int",  # Need int to save into 0, 1, ... folders (tensor otherwise)
        )

    # Augment images, only need train
    logger.info("Augmenting training dataset...")
    x_train = augment_ds(x_train)

    # Save
    logger.info("Saving augmented training data...")
    path = dataset_settings.PROCESSED_DATA_PATH / f'{task_type.value}_augmented'
    save_dataset_by_label(x_train, path / "train")

    # Save non-augmented val and test for easier management during training
    logger.info("Saving validation and test data...")
    save_dataset_by_label(x_val, path / "val")
    save_dataset_by_label(x_test, path / "test")

    logger.info("[SUCCESS] Augmented datasets saved successfully")
