from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping


def train_model(model,
                *train_data: np.ndarray | Dataset,
                validation_data: tuple[np.ndarray] | Dataset,
                batch_size: int = 32,
                epochs: int = 50,
                patience: int = 5,
                class_weights: dict = None,
                ) -> tuple[dict, keras.Model]:
    """Trains the model and
    returns the training history and the trained model

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

    Returns:
        history: training history (loss, accuracy, etc.)
        model: the trained model
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',  # monitors loss on validation
        patience=patience,  # stops if no improvement after [patience] epochs
        restore_best_weights=True  # restores weights from best epoch
    )

    history = model.fit(
        *train_data,  # training data
        validation_data=validation_data,  # validation data
        epochs=epochs,  # maximum 50 epochs
        batch_size=batch_size,  # 32 images per batch
        callbacks=[early_stopping],  # automatically stops if plateau
        class_weight=class_weights,
    )
    return history, model


def get_classification_report(
        model: keras.Model,
        *validation_data: np.ndarray | Dataset,
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
    if len(validation_data) == 1 and isinstance(validation_data[0], Dataset):
        # Need to extract y_val as np.array for sklearn classification report
        validation_images = []
        labels = []

        # Only iterate ONCE ! Each iteration shuffles the dataset.
        for tensor, label in validation_data[0].as_numpy_iterator():
            validation_images.append(tensor)
            labels.append(label)

        # Iterator returns by batch, need concatenation to removed batch
        y_val = np.concatenate(labels, axis=0)
        x_val = np.concatenate(validation_images, axis=0)

    elif len(validation_data) == 1:
        # Consider 2 args X_train and y_val as np.array
        x_val, y_val = validation_data

    else:
        raise ValueError("Need either a tf.Dataset with labels or np.ndarray x_val, y_val !")

    # Model returns a probability of class 1 => round
    y_pred = np.round(model.predict(x_val), 0)

    return classification_report(y_val, y_pred)


def plot_history(history):
    """Displays accuracy and loss curves train vs validation

    Args:
        history : history returned by model.fit()
    """
    print("📊 Generating training curves...")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ─────────────────────────────────────────
    axes[0].plot(history.history['accuracy'], label='Train')      # train curve
    axes[0].plot(history.history['val_accuracy'], label='Validation')  # val curve
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ─────────────────────────────────────────────
    axes[1].plot(history.history['loss'], label='Train')          # train curve
    axes[1].plot(history.history['val_loss'], label='Validation')     # val curve
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Progress', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✅ Curves displayed")


def get_class_weights(dataset: Dataset, encoded: bool = False) -> dict:
    # Extract class labels
    class_labels = []
    for _, labels in dataset.unbatch():
        if encoded:
            # Convert one-hot label to class index
            labels = tf.argmax(labels, axis=-1)
        class_labels.append(labels.numpy())

    # Compute class weights
    counter = Counter(class_labels)
    max_count = float(max(counter.values()))
    class_weights = {class_id: max_count / count for class_id, count in counter.items()}
    return class_weights
