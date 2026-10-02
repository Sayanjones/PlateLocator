"""Inference helpers shared by the CLI and the Streamlit app.

    python -m src.predict path/to/car.png
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

from .data import IMAGE_SIZE, preprocess

DEFAULT_MODEL = "models/car_plate_detector.keras"


class PlateDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        # compile=False: we only need predictions, not the custom IoU metric
        self.model = load_model(model_path, compile=False)

    def predict_box(self, image: Image.Image):
        """Return (xmin, ymin, xmax, ymax) in the pixel coordinates of `image`."""
        image = image.convert("RGB")  # drops alpha channel from PNGs, keeps RGB order
        w, h = image.size
        small = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype="uint8")[None]
        xmin, ymin, xmax, ymax = self.model.predict(preprocess(small), verbose=0)[0]
        box = np.array([xmin * w, ymin * h, xmax * w, ymax * h])
        box = np.clip(box, 0, [w, h, w, h])
        x1, x2 = sorted((box[0], box[2]))
        y1, y2 = sorted((box[1], box[3]))
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def draw(image: Image.Image, box, color=(0, 255, 0), width=3):
        out = image.convert("RGB").copy()
        ImageDraw.Draw(out).rectangle(box, outline=color, width=width)
        return out

    @staticmethod
    def crop(image: Image.Image, box):
        return image.convert("RGB").crop(box)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="prediction.png")
    a = ap.parse_args()
    det = PlateDetector(a.model)
    img = Image.open(a.image)
    box = det.predict_box(img)
    det.draw(img, box).save(a.out)
    print(f"box (xmin, ymin, xmax, ymax): {box} -> saved {a.out}")
