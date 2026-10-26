"""Sample-image grid and class distribution figure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import data


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = data.load_arrays()
    rng = np.random.RandomState(C.SEED)

    # one random training example per class
    fig, axes = plt.subplots(5, 9, figsize=(13, 8))
    for cls, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if cls < C.NUM_CLASSES:
            idx = rng.choice(np.where(d["train_y"] == cls)[0])
            ax.imshow(d["train_x"][idx])
            ax.set_title(f"class {cls}", fontsize=8)
    fig.suptitle("GTSRB samples at 64x64 (one per class)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "sample_grid.png", dpi=110)
    plt.close(fig)

    # class counts: shows the imbalance (some classes have ~10x more images)
    train_counts = np.bincount(d["train_y"], minlength=C.NUM_CLASSES)
    test_counts = np.bincount(d["test_y"], minlength=C.NUM_CLASSES)
    width = 0.4
    x = np.arange(C.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width / 2, train_counts, width, label="train")
    ax.bar(x + width / 2, test_counts, width, label="test")
    ax.set_xlabel("class id")
    ax.set_ylabel("images")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "class_distribution.png", dpi=110)
    plt.close(fig)

    is_val = data.split_by_track(d["train_y"], d["train_track"])
    print("train", (~is_val).sum(), "val", is_val.sum(), "test", len(d["test_y"]))
    print("min/max train class count", train_counts.min(), train_counts.max())


if __name__ == "__main__":
    main()
