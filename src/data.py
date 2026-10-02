"""Dataset loading and preprocessing.

The preprocessing function lives here, in one place, and is used by training,
evaluation and the Streamlit app. The original project preprocessed images two
different ways (BGR from cv2 in training, RGB from PIL in the app), which
quietly hurt predictions.
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = 224


def preprocess(rgb_batch: np.ndarray) -> np.ndarray:
    """RGB uint8 array (N, H, W, 3) -> VGG16-ready float32 array."""
    return preprocess_input(rgb_batch.astype("float32"))


def parse_annotation(xml_path: str):
    """Read one Pascal VOC file.

    Returns (image_filename, [xmin, ymin, xmax, ymax]) with coordinates
    normalised to 0-1 by the ORIGINAL image size. If an image has several
    plates, the largest box is used because this model predicts a single box.
    """
    root = ET.parse(xml_path).getroot()
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    filename = root.find("filename").text

    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((xmin, ymin, xmax, ymax))
    if not boxes:
        raise ValueError(f"No plate annotated in {xml_path}")

    xmin, ymin, xmax, ymax = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return filename, [xmin / width, ymin / height, xmax / width, ymax / height]


def load_dataset(data_dir: str = "data"):
    """Load images and boxes. Expects data/images/*.png and data/annotations/*.xml.

    Images are matched to annotations through the filename stored inside each
    XML file, not by sort order.
    """
    images_dir = os.path.join(data_dir, "images")
    ann_dir = os.path.join(data_dir, "annotations")
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    if not xml_files:
        raise FileNotFoundError(
            f"No annotation files in {ann_dir}. See the Dataset section of the README."
        )

    images, boxes, names = [], [], []
    for xml_path in xml_files:
        filename, box = parse_annotation(xml_path)
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            stem = os.path.splitext(os.path.basename(xml_path))[0]
            matches = glob.glob(os.path.join(images_dir, stem + ".*"))
            if not matches:
                continue
            img_path = matches[0]
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(img, dtype="uint8"))
        boxes.append(box)
        names.append(os.path.basename(img_path))

    return np.stack(images), np.asarray(boxes, dtype="float32"), names


def split_dataset(X, y, names):
    """Same split every time: ~81% train, 9% val, 10% test."""
    idx = np.arange(len(X))
    train_val, test = train_test_split(idx, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=1)
    pick = lambda ids: (X[ids], y[ids], [names[i] for i in ids])
    return pick(train), pick(val), pick(test)
