"""Baitwatch — Bounding Box Pipeline
build_bbox_dataframe : parses YOLO labels → DataFrame
crop_bb : crops bounding boxes from images
reshape_pad_crop : resizes + pads crops to target format
"""

import cv2 as cv
import pandas as pd
import tensorflow as tf

from baitwatch.settings import dataset_settings


def build_bbox_dataframe(
        labels_dataset: tf.data.Dataset,
        img_size: tuple[int, int] = dataset_settings.ORIGINAL_SIZE,
) -> pd.DataFrame:
    """Reads YOLO label files and returns a DataFrame
    with the pixel coordinates of each bounding box.
    """
    # BE CAREFUL EXPECT IMG SIZE TO BE IN TENSORFLOW FORMAT
    height, width = img_size
    rows = []

    print("📄 Reading YOLO's labels...")

    for idx, txt in enumerate(labels_dataset.as_numpy_iterator()):
        txt = txt.decode("utf-8").strip()
        if txt == "":
            continue
        for line in txt.split("\n"):
            parts = line.split(" ")
            class_id = int(parts[0])
            center_x = float(parts[1]) * width
            center_y = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height
            rows.append({
                "file_idx": idx,
                "class_id": class_id,
                "center_x": center_x,
                "center_y": center_y,
                "width": w,
                "height": h,
                "area": w * h
            })

    print(f"✅ {len(rows)} bounding boxes extracted from {idx + 1} label files")

    return pd.DataFrame(rows)


def crop_bb(labels_bb_df, img_dataset):
    """Crop each bounding box from the images.
    Returns the list of crops (np.array) and their class_id.
    """
    cropped_img = []
    class_bb = []
    img_df = []

    print("✂️  Loading images into memory...")

    for ten in img_dataset:
        img_df.append(ten.numpy())

    print(f"   {len(img_df)} images loaded")
    print("🔲 Cropping bounding boxes...")

    for bb in range(len(labels_bb_df)):
        num_img = labels_bb_df.iloc[bb]['file_idx']
        img_with_bb = img_df[int(num_img)]
        bb_label = labels_bb_df.iloc[bb]
        center_x = bb_label.loc["center_x"]
        center_y = bb_label.loc["center_y"]
        width = bb_label.loc["width"]
        height = bb_label.loc["height"]

        bounding_box = img_with_bb[
            int(center_y - height / 2): int(center_y + height / 2) + 1,
            int(center_x - width / 2): int(center_x + width / 2) + 1, :]

        cropped_img.append(bounding_box)
        class_bb.append(int(labels_bb_df.iloc[bb]["class_id"]))

    print(f"✅ {len(cropped_img)} crops generated")

    return cropped_img, class_bb


def reshape_pad_crop(cropped_img, format_img):
    """Resize each crop while keeping the aspect ratio,
    then pad to reach the target format (h, w).
    """
    bb_crop_fin = []

    print(f"📐 Resize + pad crops to {format_img}...")

    for img in cropped_img:
        img_proc = img.astype("uint8")

        ratio = max(img_proc.shape[0] / format_img[0], img_proc.shape[1] / format_img[1])

        img_resize = cv.resize(img_proc, (int(img_proc.shape[1] / ratio),
                                          int(img_proc.shape[0] / ratio)))

        bb_crop_fin.append(tf.image.pad_to_bounding_box(img_resize, format_img[0] - img_resize.shape[0],
                                                        format_img[1] - img_resize.shape[1],
                                                        format_img[0],
                                                        format_img[1]))

    print(f"✅ {len(bb_crop_fin)} crops resized to {format_img}")

    return bb_crop_fin
