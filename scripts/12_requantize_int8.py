"""Step 2 of the ORT-saturation fix: re-quantize the six INT8 variants with reduce_range=True.

teacher, student and distilled student, each plain (500 clean calibration images) and weather-calibrated
(250 clean + 250 fog/rain/low-light images): the same calibration sets as before, only the weight setting
changes. The old files were renamed to *_ortdefault.onnx beforehand and stay as a separate group.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import corruptions, data, onnx_utils

ONNX_DIR = C.CKPT_DIR / "onnx"


def main():
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    train_x = d["train_x"][~is_val]
    clean = onnx_utils.calibration_images(train_x)
    weather = corruptions.weather_calibration_set(train_x)
    for stem in ("teacher", "student", "student_kd"):
        for tag, calib in (("int8", clean), ("int8_wcal", weather)):
            out = ONNX_DIR / f"{stem}_{tag}.onnx"
            if not out.exists():
                onnx_utils.quantize_int8(ONNX_DIR / f"{stem}_fp32.onnx", out, calib)
            print("built", out.name, flush=True)


if __name__ == "__main__":
    main()
