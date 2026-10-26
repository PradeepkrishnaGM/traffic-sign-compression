"""Step 1 of the ORT-saturation fix: does reduce_range=True repair the plain INT8 student?

Same calibration images and settings as the plain INT8 student (U8S8, QDQ, per-channel), only with
reduce_range=True (weights use 7 bits, which keeps ONNX Runtime's AVX2 U8S8 kernel from saturating).
Compared with FP32, the old INT8 model and the U8U8 model from the fog follow-up.
Writes results/reduce_range_test.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from src import config as C
from src import benchmark, data, onnx_utils

ONNX_DIR = C.CKPT_DIR / "onnx"


def main():
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    out = ONNX_DIR / "student_int8_reducerange_test.onnx"
    quantize_static(str(ONNX_DIR / "student_fp32_pre.onnx"), str(out),
                    onnx_utils.CalibrationReader(onnx_utils.calibration_images(d["train_x"][~is_val])),
                    quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    models = [("fp32", "student_fp32"), ("int8_u8s8_default", "student_int8"),
              ("int8_u8u8", "student_int8_u8u8"), ("int8_u8s8_reduce_range", "student_int8_reducerange_test")]
    sample = np.array(np.load(C.CACHE_DIR / "test_x.npy", mmap_mode="r")[:1])
    rows = []
    for label, stem in models:                      # latency first, while nothing else is running
        med, p90 = benchmark.ort_latency(ONNX_DIR / f"{stem}.onnx", sample)
        rows.append({"setting": label, "file": stem, "size_MB": benchmark.file_size_mb(ONNX_DIR / f"{stem}.onnx"),
                     "latency_ms_cpu": med, "latency_p90_ms": p90})
    sets = {"clean": d["test_x"]}
    for s in C.SEVERITIES:
        sets[f"fog_{s}"] = np.load(C.CACHE_DIR / "corrupt" / f"fog_{s}.npy", mmap_mode="r")
    for row in rows:
        sess = onnx_utils.make_session(ONNX_DIR / f"{row['file']}.onnx", C.NUM_THREADS)
        for name, images in sets.items():
            row[name] = float((onnx_utils.ort_logits(sess, images).argmax(1) == d["test_y"]).mean())
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "reduce_range_test.csv", index=False)


if __name__ == "__main__":
    main()
