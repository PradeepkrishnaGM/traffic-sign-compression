"""All tunable numbers live here, with the reason for each default.

Change a value here (not inside a script) so every experiment stays comparable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gtsrb"
TRAIN_DIR = DATA_DIR / "GTSRB" / "Training"
TEST_DIR = DATA_DIR / "GTSRB" / "Final_Test" / "Images"
TEST_CSV = DATA_DIR / "GT-final_test.csv"

CACHE_DIR = ROOT / "cache"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# ---- data -------------------------------------------------------------
NUM_CLASSES = 43
IMG_SIZE = 64          # same for teacher and student so corruptions are identical
SEED = 0               # one seed for everything that is random
VAL_TRACK_FRACTION = 0.10   # share of tracks per class held out for validation

# ImageNet statistics: the teacher was pretrained with them, the student uses the same for simplicity.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---- augmentation (no flip: it would change the meaning of some signs) ---
AUG_ROTATION_DEG = 10
AUG_TRANSLATE_FRAC = 0.08
AUG_BRIGHTNESS = 0.3
AUG_CONTRAST = 0.3

# ---- training ---------------------------------------------------------
BATCH_SIZE = 128
TEACHER_EPOCHS = 10
TEACHER_LR = 0.01        # SGD with momentum; small lr because weights are pretrained
STUDENT_EPOCHS = 30
STUDENT_LR = 1e-3        # Adam
WEIGHT_DECAY = 1e-4
NUM_THREADS = 6          # physical cores; hyper-threads rarely help conv training

# ---- pruning ----------------------------------------------------------
PRUNE_AMOUNTS = (0.5, 0.7, 0.9)
PRUNE_FT_EPOCHS = 3
PRUNE_FT_LR = 0.003

# ---- distillation -----------------------------------------------------
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7

# ---- quantization -----------------------------------------------------
CALIB_IMAGES = 500
# 7-bit weights: avoids the saturation of ONNX Runtime's U8S8 kernel on CPUs without VNNI (see docs)
INT8_REDUCE_RANGE = True
WCAL_CLEAN = 250                         # weather-aware calibration: half clean ...
WCAL_CORRUPTIONS = ("fog", "rain", "low_light")   # ... half from these seen corruptions
HELDOUT_CORRUPTIONS = ("motion_blur", "gaussian_noise", "jpeg")

# ---- corruptions ------------------------------------------------------
CORRUPTIONS = ("fog", "rain", "low_light", "motion_blur", "gaussian_noise", "jpeg")
SEVERITIES = (1, 2, 3, 4, 5)
CORRUPTION_SEED = 1234

# ---- analysis ---------------------------------------------------------
ECE_BINS = 15

# ---- latency ----------------------------------------------------------
LAT_WARMUP = 50
LAT_RUNS = 200
LAT_THREADS = 1          # one thread mimics a small embedded CPU core

# ---- exploratory fog follow-up (scripts/07, 08) ----------------------------
FIRST_CONV_NODE = "/features/features.0/features.0.0/Conv"   # first conv of the student CNN
FC_NODE = "/fc/Gemm"                                          # final linear layer
LAYER_ERROR_IMAGES = 500
