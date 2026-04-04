"""IFSP model."""

from tensorflow import keras
from tensorflow.keras import layers

from baitwatch.settings import ifsp_settings


def build_model() -> keras.models.Model:
    """Build a CNN model for IFSP task.

    Returns:
        keras.models.Model: Built model
    """
    # Input layer
    img_size = ifsp_settings.CROP_IMG_SIZE
    inputs = keras.Input(shape=(*img_size, 3))

    # Normalize images
    x = keras.layers.Rescaling(scale=1.0 / 255)(inputs)

    # Hidden layers Conv
    x = layers.Conv2D(
        32,
        kernel_size=3,
        kernel_initializer="he_uniform",
        bias_initializer="ones",
        padding="same",
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        kernel_initializer="he_uniform",
        bias_initializer="ones",
        padding="same",
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        128,
        kernel_size=3,
        kernel_initializer="he_uniform",
        bias_initializer="ones",
        padding="same",
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer="he_uniform",
        bias_initializer="ones",
        padding="same",
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        512,
        kernel_size=3,
        kernel_initializer="he_uniform",
        bias_initializer="ones",
        padding="same",
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # Hidden layers Dense
    x = layers.Flatten()(x)  # aplatit en 1D
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)

    # Output layer
    outputs = layers.Dense(8, "softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


def get_optimizer() -> keras.optimizers.Optimizer:
    """Optimizer for IFSP training.

    Returns:
        keras.optimizers.Optimizer: Optimizer for IFSP training
    """
    # Use an adaptative learning rate to ensure reliability of train
    lr = keras.optimizers.schedules.ExponentialDecay(0.0003, 200, 0.96)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def compile_model(model: keras.Model, optimizer: keras.optimizers.Optimizer) -> keras.Model:
    """Compile IFSP model.

    Use Accuracy, Recall, Precision and AUC as metrics.

    Args:
        model (keras.Model): Model to compile
        optimizer (keras.optimizers.Optimizer): Optimizer to use

    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "recall", "precision", "AUC"],
    )
    return model
