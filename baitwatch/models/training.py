"""Baitwatch — Model Training Utils."""

from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping

from baitwatch.logger import logger


def train_model(
    model: keras.Model,  # noqa: PLR0913
    *train_data: np.ndarray | Dataset,
    validation_data: tuple[np.ndarray, np.ndarray] | Dataset,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 5,
    class_weights: dict | None = None,
) -> tuple[keras.callbacks.History, keras.Model]:
    """Trains the model and returns the training history and the trained model.

    Usage:
        >>> history, model = train_model(model, X_train, y_train, validation_data=(X_val, y_val))

        >>> history, model = train_model(model, X_train_dataset, validation_data=X_val_dataset)

    Args:
        model: the model to train
        train_data: training data
        validation_data: validation data
        batch_size: batch size
        epochs: maximum number of epochs
        patience: number of epochs without improvement before stopping
        class_weights: class weights for imbalanced datasets (optional)

    Returns:
        history: training history (loss, accuracy, etc.).
        model: the trained model.
    """
    logger.info(
        f"Starting model training with batch_size={batch_size}, epochs={epochs},"
        f" patience={patience}"
    )
    if class_weights:
        logger.debug(f"Using class weights: {class_weights}")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    history = model.fit(
        *train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    final_epoch = len(history.history["loss"])
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    logger.info(f"Training completed after {final_epoch} epochs")
    logger.info(f"Final training loss: {final_loss:.4f}, validation loss: {final_val_loss:.4f}")

    return history, model


def get_classification_report(
    model: keras.Model,
    validation_data: tuple[np.ndarray, np.ndarray] | Dataset,
) -> str:
    """Return classification report based on given validation data and model.

    Usage:
        >>> report = get_classification_report(model, dataset)  # With tf.Dataset including labels

        >>> report = get_classification_report(model, X_train, y_train)  # With Numpy arrays

    Args:
        model: keras model to evaluate
        validation_data: either a tf.Dataset with labels or np.ndarray x_val, y_val
                         containing data to get classification report from

    Returns:
        classification report as strings (to be printed)
    """
    logger.debug("Generating classification report")
    if isinstance(validation_data, Dataset):
        # Need to extract y_val as np.array for sklearn classification report
        validation_images = []
        labels = []

        # Only iterate ONCE ! Each iteration shuffles the dataset.
        for tensor, label in validation_data.as_numpy_iterator():
            validation_images.append(tensor)
            labels.append(label)

        # Iterator returns by batch, need concatenation to removed batch
        y_val = np.concatenate(labels, axis=0)
        x_val = np.concatenate(validation_images, axis=0)

    else:
        # Consider 2 args X_train and y_val as np.array
        x_val, y_val = validation_data

    # Model returns a probability of class 1 => round
    logger.debug(f"Predicting on {len(x_val)} validation samples")
    y_pred = np.round(model.predict(x_val), 0)

    logger.info("Classification report generated successfully")
    return classification_report(y_val, y_pred)


def plot_history(history: keras.callbacks.History) -> None:
    """Displays accuracy and loss curves train vs validation.

    Args:
        history : history returned by model.fit()
    """
    logger.info("[PLOT] Generating training curves...")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ─────────────────────────────────────────
    axes[0].plot(history.history["accuracy"], label="Train")  # train curve
    axes[0].plot(history.history["val_accuracy"], label="Validation")  # val curve
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ─────────────────────────────────────────────
    axes[1].plot(history.history["loss"], label="Train")  # train curve
    axes[1].plot(history.history["val_loss"], label="Validation")  # val curve
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Progress", fontsize=14)
    plt.tight_layout()
    plt.show()

    logger.info("[SUCCESS] Curves displayed")


def get_class_weights(dataset: Dataset, encoded: bool = False) -> dict:
    """Compute class weights from imbalanced dataset based on inverse frequency.

    Args:
        dataset (Dataset): Dataset to compute class weights from
        encoded (bool): Whether labels are encoded (one-hot)

    Returns:
        dict: Class weights
    """
    logger.debug("Computing class weights from dataset")
    # Extract class labels
    class_labels = []
    for _, labels in dataset.unbatch():
        # Convert one-hot label to class index
        label = tf.argmax(labels, axis=-1) if encoded else labels
        class_labels.append(label.numpy())

    # Compute class weights
    counter = Counter(class_labels)
    max_count = float(max(counter.values()))
    class_weights = {class_id: max_count / count for class_id, count in counter.items()}
    logger.info(f"Class distribution: {dict(counter)}")
    logger.debug(f"Computed class weights: {class_weights}")
    return class_weights
