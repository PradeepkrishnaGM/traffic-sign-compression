"""Dataset loading: decode once, resize to 64x64, cache as uint8 .npy, split by track."""
import numpy as np
import cv2
import pandas as pd
import torch

from . import config as C


def _read_rgb(path):
    """Read an image file and return an RGB uint8 array resized to IMG_SIZE x IMG_SIZE."""
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # INTER_AREA is the right choice when shrinking (it averages pixels, so fewer aliasing artefacts).
    return cv2.resize(rgb, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)


def build_cache():
    """Decode every .ppm once. Later runs just load the .npy files (much faster)."""
    C.CACHE_DIR.mkdir(exist_ok=True)
    if (C.CACHE_DIR / "test_x.npy").exists():
        return
    images, labels, tracks = [], [], []
    for class_dir in sorted(C.TRAIN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.ppm")):
            images.append(_read_rgb(path))
            labels.append(int(class_dir.name))
            # file name is <track>_<frame>.ppm; the track id is only unique inside a class
            tracks.append(int(path.stem.split("_")[0]))
    np.save(C.CACHE_DIR / "train_x.npy", np.stack(images))
    np.save(C.CACHE_DIR / "train_y.npy", np.array(labels, dtype=np.int64))
    np.save(C.CACHE_DIR / "train_track.npy", np.array(tracks, dtype=np.int64))

    table = pd.read_csv(C.TEST_CSV, sep=";")
    test_images = [_read_rgb(C.TEST_DIR / name) for name in table["Filename"]]
    np.save(C.CACHE_DIR / "test_x.npy", np.stack(test_images))
    np.save(C.CACHE_DIR / "test_y.npy", table["ClassId"].to_numpy(dtype=np.int64))


def load_arrays():
    """Return dict with train_x/y/track and test_x/y (uint8 NHWC images)."""
    build_cache()
    names = ["train_x", "train_y", "train_track", "test_x", "test_y"]
    return {n: np.load(C.CACHE_DIR / f"{n}.npy") for n in names}


def split_by_track(labels, tracks, fraction=C.VAL_TRACK_FRACTION, seed=C.SEED):
    """Return boolean mask: True = validation.

    Frames of one physical sign (one track) look almost identical, so a random
    split would put near-copies in both train and val and inflate val accuracy.
    We hold out whole tracks instead, about `fraction` of the tracks per class.
    """
    rng = np.random.RandomState(seed)
    is_val = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        in_class = labels == cls
        class_tracks = np.unique(tracks[in_class])
        n_val = max(1, int(round(fraction * len(class_tracks))))
        val_tracks = rng.choice(class_tracks, size=n_val, replace=False)
        is_val |= in_class & np.isin(tracks, val_tracks)
    return is_val


def to_tensor(images_uint8):
    """uint8 NHWC -> normalised float32 NCHW tensor (same preprocessing for every model)."""
    x = torch.from_numpy(images_uint8).float().div_(255.0).permute(0, 3, 1, 2)
    mean = torch.tensor(C.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(C.STD).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous()


def _augment_batch(x, generator):
    """Small random rotation + translation + brightness/contrast on a float [0,1] NCHW batch.

    Done with one affine grid per batch element so it stays pure PyTorch and
    reproducible through `generator`. No flip, no weather effects.
    """
    n = x.shape[0]
    angle = (torch.rand(n, generator=generator) * 2 - 1) * C.AUG_ROTATION_DEG * np.pi / 180
    shift = (torch.rand(n, 2, generator=generator) * 2 - 1) * C.AUG_TRANSLATE_FRAC * 2
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    theta[:, :, 2] = shift
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="border", align_corners=False)
    brightness = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * C.AUG_BRIGHTNESS
    contrast = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * C.AUG_CONTRAST
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean * brightness
    return x.clamp(0, 1)


def batches(images, labels, batch_size, train, seed=0):
    """Yield (normalised tensor, label tensor) batches. `train=True` shuffles and augments."""
    n = len(labels)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=gen).numpy() if train else np.arange(n)
    mean = torch.tensor(C.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(C.STD).view(1, 3, 1, 1)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        x = torch.from_numpy(images[idx]).float().div_(255.0).permute(0, 3, 1, 2)
        if train:
            x = _augment_batch(x, gen)
        x = (x - mean) / std
        yield x.contiguous(), torch.from_numpy(labels[idx])
