"""Exploratory check: does the INT8 accuracy loss come from ORT's fused integer kernels?

The same INT8 file is run twice: with ONNX Runtime's default graph optimisations (QDQ pairs are fused
into integer kernels such as QLinearConv) and with all optimisations disabled (the model then runs as
plain float maths between quantize/dequantize nodes, which is what the fake-quantization of the
layer-error diagnostic reproduces). Results: results/ort_fusion_check.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import pandas as pd

from src import config as C
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from src import data, onnx_utils

ONNX_DIR = C.CKPT_DIR / "onnx"
MODELS = ["student_int8", "student_kd_int8", "student_int8_fp32first", "teacher_int8", "student_int8_u8u8"]


def build_u8u8(train_x):
    """Same INT8 recipe, but weights stored as UNSIGNED 8-bit like the activations (U8U8).

    ONNX Runtime then uses a different integer kernel path. This tests whether the accuracy loss of the
    default signed-weight (U8S8) model comes from that kernel path (a known AVX2 saturation issue).
    """
    out = ONNX_DIR / "student_int8_u8u8.onnx"
    if out.exists():
        return
    quantize_static(str(ONNX_DIR / "student_fp32_pre.onnx"), str(out),
                    onnx_utils.CalibrationReader(onnx_utils.calibration_images(train_x)),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)


def session(path, optimised):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = C.NUM_THREADS
    if not optimised:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def main():
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    build_u8u8(d["train_x"][~is_val])
    sets = {"clean": d["test_x"], "fog_5": np.load(C.CACHE_DIR / "corrupt" / "fog_5.npy", mmap_mode="r")}
    rows = []
    for name in MODELS:
        for mode, optimised in (("default_optimised", True), ("optimisations_off", False)):
            sess = session(ONNX_DIR / f"{name}.onnx", optimised)
            row = {"model": name, "mode": mode}
            for cond, images in sets.items():
                pred = onnx_utils.ort_logits(sess, images).argmax(1)
                row[cond] = float((pred == d["test_y"]).mean())
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ort_fusion_check.csv", index=False)


if __name__ == "__main__":
    main()
