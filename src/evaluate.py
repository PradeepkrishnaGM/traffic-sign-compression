"""Accuracy helpers shared by all scripts."""
import numpy as np

from . import config as C


def per_class_accuracy(pred, labels):
    """Accuracy for each of the 43 classes (nan if a class is absent)."""
    out = np.full(C.NUM_CLASSES, np.nan)
    for cls in range(C.NUM_CLASSES):
        mask = labels == cls
        if mask.any():
            out[cls] = float((pred[mask] == cls).mean())
    return out
