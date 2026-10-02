"""Evaluate on the held-out test split and save a prediction grid.

    python -m src.evaluate --data-dir data
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from .data import load_dataset, preprocess, split_dataset
from .predict import DEFAULT_MODEL


def box_iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: np.clip(r[:, 2] - r[:, 0], 0, None) * np.clip(r[:, 3] - r[:, 1], 0, None)
    return inter / (area(a) + area(b) - inter + 1e-7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    X, y, names = load_dataset(args.data_dir)
    _, _, (Xte, yte, nte) = split_dataset(X, y, names)
    model = load_model(args.model, compile=False)
    pred = model.predict(preprocess(Xte), verbose=0)

    ious = box_iou(yte, pred)
    print(f"test images: {len(Xte)}")
    print(f"mean IoU: {ious.mean():.3f} | median IoU: {np.median(ious):.3f}")
    for t in (0.5, 0.75):
        print(f"IoU >= {t}: {(ious >= t).mean() * 100:.1f}%")

    n = min(12, len(Xte))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, img, gt, pr, iou in zip(axes.ravel(), Xte[:n], yte[:n], pred[:n], ious[:n]):
        s = img.shape[0]
        ax.imshow(img); ax.axis("off"); ax.set_title(f"IoU {iou:.2f}", fontsize=9)
        for box, color in ((gt, "lime"), (pr, "red")):
            ax.add_patch(plt.Rectangle((box[0] * s, box[1] * s), (box[2] - box[0]) * s,
                                       (box[3] - box[1]) * s, fill=False, edgecolor=color, linewidth=2))
    fig.suptitle("Green = ground truth, red = prediction (test split)")
    fig.tight_layout(); fig.savefig("assets/results/test_predictions.png", dpi=110)
    print("saved assets/results/test_predictions.png")


if __name__ == "__main__":
    main()
