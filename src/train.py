"""Train the plate detector.

    python -m src.train --data-dir data
    python -m src.train --data-dir data --finetune
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from .data import load_dataset, preprocess, split_dataset
from .model import build_model, compile_model


def plot_history(histories, path):
    loss = sum((h.history["loss"] for h in histories), [])
    val_loss = sum((h.history["val_loss"] for h in histories), [])
    iou = sum((h.history["val_iou_metric"] for h in histories), [])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(loss, label="train"); ax[0].plot(val_loss, label="val")
    ax[0].set_title("MSE loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(iou, color="green"); ax[1].set_title("Validation IoU"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--finetune", action="store_true", help="after the frozen stage, unfreeze VGG16 block5 and train a bit more")
    p.add_argument("--out", default="models/car_plate_detector.keras")
    args = p.parse_args()

    tf.keras.utils.set_random_seed(42)
    weights = None if args.weights.lower() == "none" else args.weights

    X, y, names = load_dataset(args.data_dir)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = split_dataset(X, y, names)
    print(f"images: {len(X)} | train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")
    Xtr, Xva, Xte = preprocess(Xtr), preprocess(Xva), preprocess(Xte)

    model, base = build_model(weights=weights, freeze_base=True)
    compile_model(model, 1e-3)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    histories = [model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs,
                           batch_size=args.batch_size, callbacks=[stop], verbose=2)]

    if args.finetune:
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
        compile_model(model, 1e-5)  # recompile after changing trainable flags
        stop_ft = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        histories.append(model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30,
                                   batch_size=args.batch_size, callbacks=[stop_ft], verbose=2))

    test_loss, test_iou = model.evaluate(Xte, yte, verbose=0)
    print(f"test MSE {test_loss:.5f} | test mean IoU {test_iou:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    os.makedirs("assets/results", exist_ok=True)
    plot_history(histories, "assets/results/training_curves.png")
    with open("assets/results/metrics.json", "w") as f:
        json.dump({"test_mse": float(test_loss), "test_mean_iou": float(test_iou),
                   "finetuned": args.finetune, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte)}, f, indent=2)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
