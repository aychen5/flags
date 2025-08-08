# utils.py
import os
import pickle
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import pandas as pd
import boto3
from botocore.config import Config


import tensorflow as tf
from tensorflow import keras
import keras_cv #pip install --upgrade git+https://github.com/keras-team/keras-cv -q
from keras import mixed_precision
from keras.callbacks import TensorBoard
from keras_cv import bounding_box, visualization


# --- AWS + S3 Utilities ---
def setup_s3_client(access_key, secret_key, max_connections=100):
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    config = Config(max_pool_connections=max_connections)
    return boto3.client("s3", config=config)

def read_from_s3(s3_client, bucket_name, object_key, object_type, as_bytes=False):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read()
        f = BytesIO(content)

        if object_type == "CSV":
            return pd.read_csv(f)
        elif object_type == "XML":
            return BeautifulSoup(f, 'xml')
        elif object_type == "JPG":
            return content if as_bytes else Image.open(f)
        else:
            return content
    except Exception as e:
        print(f"❌ Problem reading {object_key}: {e}")
        return None


# --- Annotation Parsing ---
def parse_xml(xml_file):
    annotations_df = pd.DataFrame()
    for child in xml_file.find_all():
        tag = child.name
        if tag == "image":
            class_id = None
        elif tag == "box":
            class_id = 0

        if tag == "box":
            image_parent = child.find_parent()
            result = {
                "parent_tag": [tag],
                "class": [class_id],
                "id": [image_parent.get("id")],
                "filename": [image_parent.get("name")],
                "xtl": [child.get("xtl")],
                "ytl": [child.get("ytl")],
                "xbr": [child.get("xbr")],
                "ybr": [child.get("ybr")],
            }
            annotations_df = pd.concat([annotations_df, pd.DataFrame(result)])
        elif tag == "image" and list(child) == ['\n']:
            result = {
                "parent_tag": [tag],
                "class": [class_id],
                "id": [child.get("id")],
                "filename": [child.get("name")],
                "xtl": [None],
                "ytl": [None],
                "xbr": [None],
                "ybr": [None],
            }
            annotations_df = pd.concat([annotations_df, pd.DataFrame(result)])

    return annotations_df.set_index("id")


# --- Dataset Serialization ---
def save_dataset_dict(path, image_paths, classes, boxes):
    data_dict = {
        "image_paths": image_paths,
        "classes": classes,
        "boxes": boxes,
    }
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)

def load_dataset_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --- TF Data Helpers ---
def load_image_from_s3(bucket_name, image_path, s3_client):
    object_key = image_path.numpy().decode("utf-8")
    bytes_image = read_from_s3(s3_client, bucket_name, object_key, "JPG", as_bytes=True)
    img = Image.open(BytesIO(bytes_image)).convert("RGB")
    return np.array(img, dtype=np.float32)

def load_dataset(image_path, classes, boxes, s3_client, bucket_name):
    image = tf.py_function(
        func=lambda path: load_image_from_s3(bucket_name, path, s3_client),
        inp=[image_path],
        Tout=tf.float32,
    )
    image.set_shape([None, None, 3])
    bounding_boxes = {
        "classes": tf.cast(classes, tf.int32),
        "boxes": tf.cast(boxes, tf.float32)
    }
    return {"images": image, "bounding_boxes": bounding_boxes}

def dict_to_tuple(sample):
    dense = tf.keras.utils.to_dense(
        {
            "boxes": sample["bounding_boxes"]["boxes"],
            "classes": sample["bounding_boxes"]["classes"],
        }
    )
    return sample["images"], dense

# --- Dataset Construction ---
def get_augmenter(bounding_box_format="xyxy"):
    """
    Returns a Keras Sequential model that performs data augmentation
    on images and their bounding boxes.
    """
    return keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=bounding_box_format),
            keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format=bounding_box_format),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format=bounding_box_format,
            ),
        ]
    )
def build_dataset(data,
                  batch_size=16,
                  augment=False,
                  shuffle=True,
                  repeat=True,
                  bounding_box_format="xyxy"):
    """
    Constructs a TensorFlow Dataset for object detection using the KerasCV format.

    Args:
        data: tf.data.Dataset yielding (image_path, classes, boxes)
        batch_size: Size of batches.
        augment: Whether to apply data augmentation.
        shuffle: Whether to shuffle the dataset.
        repeat: Whether to repeat the dataset indefinitely.
        bounding_box_format: Format of bounding boxes (default "xyxy")

    Returns:
        A tf.data.Dataset yielding (images, bounding_boxes) tuples
        suitable for training a KerasCV object detector.
    """

    # Base preprocessing
    ds = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    # shuffle before batching
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    # ragged batch to handle variable number of boxes
    ds = ds.ragged_batch(batch_size)

    # optional aug, but always resize
    if augment:
        augmenter = get_augmenter(bounding_box_format=bounding_box_format)
        ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Resize images and boxes
        resizer = keras_cv.layers.Resizing(
            height=640,
            width=640,
            pad_to_aspect_ratio=True,
            bounding_box_format=bounding_box_format
        )
        ds = ds.map(resizer, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert to dense bounding box dict
    ds = ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

    # repeat and prefetch
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
