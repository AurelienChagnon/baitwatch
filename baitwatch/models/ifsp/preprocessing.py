import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from baitwatch.models.commons.preprocessing import preprocess_ds
from baitwatch.models.ifsp.bounding_box import build_bbox_dataframe, crop_bb, reshape_pad_crop
from baitwatch.settings import dataset_settings, ifsp_settings


def process_data_ifsp(
        imgs: Dataset,
        labels: Dataset,
) -> tuple[Dataset, np.ndarray]:
    # Process images
    imgs_preprocessed = preprocess_ds(imgs)

    # Fetch bounding boxes
    bb_df = build_bbox_dataframe(labels, img_size=dataset_settings.ORIGINAL_SIZE)

    # Crop and pad images to keep only bounding boxes
    crop_imgs, y = crop_bb(bb_df, imgs_preprocessed)
    x = Dataset.from_tensor_slices(reshape_pad_crop(crop_imgs, format_img=ifsp_settings.CROP_IMG_SIZE))

    # Convert
    y = np.array(y)

    return x, y


def preprocess_ifsp(dataset: Dataset) -> Dataset:
    dataset = preprocess_ds(dataset)

    @tf.py_function(Tout=tf.uint8)  # 8bit image
    def resize(processed_img):
        processed_img = processed_img.numpy().astype("uint8")
        resized_img = reshape_pad_crop([processed_img], format_img=ifsp_settings.CROP_IMG_SIZE)
        return resized_img
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
