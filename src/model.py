"""Model definition and the IoU metric."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

from .data import IMAGE_SIZE


def iou_metric(y_true, y_pred):
    """Mean intersection-over-union for boxes in [xmin, ymin, xmax, ymax] form."""
    x1 = tf.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = tf.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = tf.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    area_t = tf.maximum(0.0, y_true[:, 2] - y_true[:, 0]) * tf.maximum(0.0, y_true[:, 3] - y_true[:, 1])
    area_p = tf.maximum(0.0, y_pred[:, 2] - y_pred[:, 0]) * tf.maximum(0.0, y_pred[:, 3] - y_pred[:, 1])
    union = area_t + area_p - inter
    return tf.reduce_mean(inter / (union + 1e-7))


def build_model(weights="imagenet", freeze_base=True):
    """VGG16 backbone + small regression head that outputs 4 values in 0-1."""
    base = VGG16(weights=weights, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(4, activation="sigmoid")(x)  # xmin, ymin, xmax, ymax
    model = models.Model(inputs, outputs, name="plate_detector")
    return model, base


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mean_squared_error",
        metrics=[iou_metric],
    )
