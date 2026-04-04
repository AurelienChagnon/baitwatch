"""Baitwatch — Bounding Box Pipeline."""

import cv2 as cv
import numpy as np
import tensorflow as tf


def extract_fish_bounding_boxes(
    image: tf.Tensor,
    label: tf.Tensor,
    target_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[int]]:
    """Extracts fishes from image using bounding box coordinates in associated YOLO label.

    Args:
        image (tf.Tensor): Image
        label (tf.Tensor): YOLO label
        target_size (tuple[int, int]): Target image size (height, width)

    Returns:
        list[np.ndarray]: List of fishes images, cropped and resized from original image.
                          Can be empty if no fish detected.
        list[int]: List of class IDs associated to each fish
    """
    cropped_fishes = []
    labels = []
    decoded_label = label.numpy().decode("utf-8").strip()

    if not decoded_label:
        # Label file is empty: no fish, no bounding box
        # Do not remove in favor of the following for loop
        # as split an empty string does not return an empty list
        return [], []

    # Retrieve image size
    height, width = image.shape[:2]

    # When label is not empty, it contains multiple lines, each representing a bounding box
    for line in decoded_label.split("\n"):
        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        # Coordinates are normalized (values between 0.0 and 1.0) relative to the image dimensions.
        parts = line.split(" ")
        class_id = int(parts[0])
        bb_center_x = float(parts[1]) * width
        bb_center_y = float(parts[2]) * height
        bb_width = float(parts[3]) * width
        bb_height = float(parts[4]) * height
        bounding_box = image.numpy()[
            int(bb_center_y - bb_height / 2) : int(bb_center_y + bb_height / 2) + 1,
            int(bb_center_x - bb_width / 2) : int(bb_center_x + bb_width / 2) + 1,
            :,
        ]
        padded_fish = padded_resize(bounding_box, target_size)
        cropped_fishes.append(padded_fish)
        labels.append(class_id)
    return cropped_fishes, labels


def padded_resize(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize the given image while keeping the aspect ratio, then pad to reach the target size.

    Args:
        image (np.ndarray): Image to resize
        target_size (tuple[int, int]): Target size (height, width)

    Returns:
        np.ndarray: Resized and padded image
    """
    img_proc = image.astype("uint8")
    height, width = target_size

    ratio = max(img_proc.shape[0] / height, img_proc.shape[1] / width)

    img_resize = cv.resize(
        img_proc,
        (int(img_proc.shape[1] / ratio), int(img_proc.shape[0] / ratio)),
        interpolation=cv.INTER_LINEAR,
    )

    padded_img = tf.image.pad_to_bounding_box(
        img_resize,
        height - img_resize.shape[0],
        width - img_resize.shape[1],
        height,
        width,
    )

    return padded_img
