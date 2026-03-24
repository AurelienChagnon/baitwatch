import tensorflow as tf


def flip_left_right(image):
    """Flips the image horizontally (left to right).

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The horizontally flipped image.
    """
    img = tf.image.flip_left_right(image)
    return img


def flip_up_down(image):
    """Flips the image vertically (up to down).

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The vertically flipped image.
    """
    img = tf.image.flip_up_down(image)
    return img


def rotate_180(image):
    """Rotates the image by 180 degrees.

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The rotated image.
    """
    img = tf.image.rot90(image, k=2)
    return img


def add_noise(image: tf.Tensor) -> tf.Tensor:
    """Adds random Gaussian noise to the image to simulate grain or low-light conditions.

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The noisy image, cast to float32 and clipped between 0 and 255.
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=25.0, dtype=tf.float32)
    return tf.clip_by_value(tf.cast(image, tf.float32) + noise, 0, 255)


def augment_images(img, label):
    """Data augmentation pipeline generating 8 variations from a single input.

    Applies geometric transformations (flips, rotations) and photometric
    transformations (brightness, contrast, saturation, noise).
    This function is designed to be used with `tf.data.Dataset.flat_map`.

    Args:
        img (tf.Tensor): The source image tensor (typically uint8).
        label (tf.Tensor): The associated label tensor (e.g., YOLO bounding box or class).
        model_type (str): The model identifier for directory naming (e.g., 'fonf').
        split_name (str): The dataset split being processed (e.g., 'train', 'val').

    Returns:
        tf.data.Dataset: A sliced dataset containing the 8 augmented (image, label) pairs.
    """
    img_lr = flip_left_right(img)
    img_ud = flip_up_down(img)
    img_180 = rotate_180(img)

    # Force cast to uint8 for each photometric transformation
    img_br = tf.cast(tf.image.random_brightness(img, max_delta=0.8), tf.uint8)
    img_ct = tf.cast(tf.image.random_contrast(img, lower=0.2, upper=2.5), tf.uint8)
    img_st = tf.cast(tf.image.random_saturation(img, lower=0.0, upper=6.0), tf.uint8)
    img_ns = tf.cast(add_noise(img), tf.uint8)
    img = tf.cast(img, tf.uint8)
    img_lr = tf.cast(img_lr, tf.uint8)
    img_ud = tf.cast(img_ud, tf.uint8)
    img_180 = tf.cast(img_180, tf.uint8)

    aug_imgs = [img, img_lr, img_ud, img_180, img_br, img_ct, img_st, img_ns]

    aug_labs = [label] * 8

    return tf.data.Dataset.from_tensor_slices((aug_imgs, aug_labs))


def augment_ds(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Augments the dataset by applying geometric transformations."""
    dataset_aug = dataset.flat_map(lambda x, y: augment_images(x, y))
    return dataset_aug
