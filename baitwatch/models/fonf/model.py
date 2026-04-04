"""Model for Fish Or No Fish (FONF) classification task."""

from tensorflow import keras
from tensorflow.keras import layers

from baitwatch.settings import fonf_settings


def build_model() -> keras.models.Model:
    """Build a CNN model for FONF task.

    Returns:
        keras.models.Model: Built model
    """
    # Input layer
    # REMEMBER Preprocess with Opencv, which reverse order of image size compared
    # to tensorflow used to load data
    img_size = fonf_settings.PREPROCESS_IMG_SIZE[::-1]
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

    x = layers.Conv2D(64, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(
        x
    )
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.Conv2D(64, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(
        x
    )
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(
        128, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones"
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.Conv2D(
        128, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones"
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # Hidden layers Dense
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation="relu")(x)

    # Output layer
    outputs = layers.Dense(1, "sigmoid")(x)

    model = keras.Model(inputs, outputs)

    return model


def get_optimizer() -> keras.optimizers.Optimizer:
    """Optimizer for FONF training.

    Returns:
        keras.optimizers.Optimizer: Optimizer for FONF training
    """
    # For fonf, use an adaptative learning rate to ensure reliability of train
    lr = keras.optimizers.schedules.ExponentialDecay(0.0003, 200, 0.96)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def compile_model(model: keras.Model, optimizer: keras.optimizers.Optimizer) -> keras.Model:
    """Model compilation for FONF.

    Use accuracy, recall, precision and AUC as metrics.

    Args:
        model (keras.Model): Model to compile
        optimizer (keras.optimizers.Optimizer): Optimizer to use

    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "recall", "precision", "AUC"],
    )
    return model
