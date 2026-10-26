"""Deterministic driving-style corruptions for 64x64 uint8 RGB images.

Six corruptions (fog, rain, low_light, motion_blur, gaussian_noise, jpeg), severity 1 to 5.
Each image gets its own random generator, seeded from (base seed, corruption, severity,
image index), so results never depend on batch order or on how many images are processed.
"""
import cv2
import numpy as np

from . import config as C

SIZE = C.IMG_SIZE


def _rng(corruption, severity, index, seed):
    """One independent, reproducible random generator per image."""
    name_id = C.CORRUPTIONS.index(corruption)
    value = ((seed * 1000003 + name_id) * 1000003 + severity) * 1000003 + int(index)
    return np.random.RandomState(value % (2 ** 32))


def _to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def fog(img, severity, rng):
    """Blend towards a bright grey haze whose density varies smoothly across the image."""
    strength = (0.15, 0.28, 0.40, 0.52, 0.65)[severity - 1]
    low_res = rng.rand(4, 4).astype(np.float32)
    # upsampling a tiny random grid gives a smooth, cloud-like density field
    field = cv2.resize(low_res, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(strength * (0.7 + 0.6 * field), 0, 0.95)[:, :, None]
    haze = np.array([215.0, 215.0, 220.0], dtype=np.float32)
    return _to_uint8(img.astype(np.float32) * (1 - alpha) + haze * alpha)


def rain(img, severity, rng):
    """Draw slanted bright streaks, slightly darken and soften the picture."""
    n_streaks = (15, 30, 50, 75, 100)[severity - 1]
    layer = np.zeros((SIZE, SIZE), dtype=np.float32)
    slant = rng.uniform(-0.3, 0.3)
    for _ in range(n_streaks):
        x0, y0 = rng.randint(0, SIZE), rng.randint(0, SIZE)
        length = rng.randint(6, 14)
        x1, y1 = int(x0 + slant * length), y0 + length
        cv2.line(layer, (x0, y0), (x1, y1), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    layer = np.clip(layer * 1.5, 0, 1)[:, :, None]
    dark = (0.95, 0.9, 0.85, 0.8, 0.75)[severity - 1]
    base = img.astype(np.float32) * dark
    base = cv2.GaussianBlur(base, (3, 3), 0.3 * severity)
    return _to_uint8(base * (1 - 0.6 * layer) + 230.0 * 0.6 * layer)


def low_light(img, severity, rng):
    """Darken with a gamma curve, then add the sensor noise a camera produces in the dark."""
    gain = (0.75, 0.62, 0.50, 0.40, 0.32)[severity - 1]
    noise_sigma = (2, 4, 6, 8, 10)[severity - 1]
    x = (img.astype(np.float32) / 255.0) ** 1.3 * gain * 255.0
    x = x + rng.normal(0, noise_sigma, x.shape)
    return _to_uint8(x)


def motion_blur(img, severity, rng):
    """Blur along a random direction (camera or vehicle movement)."""
    length = (3, 5, 7, 9, 11)[severity - 1]
    angle = rng.uniform(0, np.pi)
    kernel = np.zeros((length, length), dtype=np.float32)
    centre = (length - 1) / 2.0
    dx, dy = np.cos(angle) * centre, np.sin(angle) * centre
    p0 = (int(round(centre - dx)), int(round(centre - dy)))
    p1 = (int(round(centre + dx)), int(round(centre + dy)))
    cv2.line(kernel, p0, p1, 1.0, 1)
    kernel /= kernel.sum()
    return _to_uint8(cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT))


def gaussian_noise(img, severity, rng):
    sigma = (8, 16, 26, 38, 52)[severity - 1]
    return _to_uint8(img.astype(np.float32) + rng.normal(0, sigma, img.shape))


def jpeg(img, severity, rng):
    """Strong JPEG compression (camera / transmission artefacts). `rng` is unused: JPEG is deterministic."""
    quality = (60, 40, 25, 15, 8)[severity - 1]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


_FUNCS = {"fog": fog, "rain": rain, "low_light": low_light, "motion_blur": motion_blur,
          "gaussian_noise": gaussian_noise, "jpeg": jpeg}


def corrupt(images, corruption, severity, indices=None, seed=C.CORRUPTION_SEED):
    """Corrupt a stack of uint8 NHWC images.

    `indices` are the images' ids used for seeding (default 0..N-1). Pass the original
    dataset index when corrupting a subset so the result matches the full-set result.
    """
    if indices is None:
        indices = np.arange(len(images))
    func = _FUNCS[corruption]
    out = np.empty_like(images)
    for i, (img, idx) in enumerate(zip(images, indices)):
        out[i] = func(img, severity, _rng(corruption, severity, idx, seed))
    return out


def weather_calibration_set(train_x, n_total=C.CALIB_IMAGES, n_clean=C.WCAL_CLEAN, seed=C.SEED):
    """Calibration images for weather-aware INT8: n_clean clean + the rest corrupted.

    Only the corruptions in C.WCAL_CORRUPTIONS (fog, rain, low light) are used, with severities
    1 to 5 cycled evenly. Motion blur, noise and JPEG are never used here, so they stay a fair
    held-out test. Images come from the training split, never from the test set.
    """
    chosen = np.random.RandomState(seed).choice(len(train_x), n_total, replace=False)
    clean_idx, weather_idx = chosen[:n_clean], chosen[n_clean:]
    parts = [train_x[clean_idx]]
    for c_i, corruption in enumerate(C.WCAL_CORRUPTIONS):
        for s_i, severity in enumerate(C.SEVERITIES):
            # image j gets corruption j % 3 and severity (j // 3) % 5 + 1
            picks = [j for j in range(len(weather_idx))
                     if j % len(C.WCAL_CORRUPTIONS) == c_i and (j // len(C.WCAL_CORRUPTIONS)) % len(C.SEVERITIES) == s_i]
            if picks:
                ids = weather_idx[picks]
                parts.append(corrupt(train_x[ids], corruption, severity, indices=ids))
    return np.concatenate(parts)
