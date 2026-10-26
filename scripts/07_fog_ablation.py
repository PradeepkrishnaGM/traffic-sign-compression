"""Exploratory follow-up: which INT8 layers make the small students fail in fog?

Builds 3 extra INT8 variants per student (same 500 clean calibration images as the plain INT8
model, same settings), only with some layers excluded from quantization (kept in FP32):
  fp32first = first conv, fp32fc = final linear layer, fp32both = both.
Evaluates them on clean + 30 corruption conditions (cached corrupted sets, resumable) and writes
results/fog_ablation.csv (accuracy and size; latency is added by 08_fog_diagnostics.py).
Predictions go to cache/preds_ablation/, separate from the 12 main variants.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src import benchmark, data, onnx_utils

ONNX_DIR = C.CKPT_DIR / "onnx"
CORRUPT_DIR = C.CACHE_DIR / "corrupt"
PRED_DIR = C.CACHE_DIR / "preds"                 # main variants (read only here)
ABL_DIR = C.CACHE_DIR / "preds_ablation"

VARIANT_NODES = {
    "int8_fp32first": [C.FIRST_CONV_NODE],
    "int8_fp32fc": [C.FC_NODE],
    "int8_fp32both": [C.FIRST_CONV_NODE, C.FC_NODE],
}
STUDENTS = {"student": "student_fp32", "student_kd": "student_kd_fp32"}   # prefix -> FP32 onnx stem
CONDITIONS = ["clean"] + [f"{c}_{s}" for c in C.CORRUPTIONS for s in C.SEVERITIES]


def build_models(train_x):
    calib = onnx_utils.calibration_images(train_x)        # identical to the plain INT8 calibration set
    for prefix, fp32_stem in STUDENTS.items():
        for tag, nodes in VARIANT_NODES.items():
            out = ONNX_DIR / f"{prefix}_{tag}.onnx"
            if not out.exists():
                onnx_utils.quantize_int8(ONNX_DIR / f"{fp32_stem}.onnx", out, calib, nodes_to_exclude=nodes)


def run_inference(test_x):
    for prefix in STUDENTS:
        for tag in VARIANT_NODES:
            variant = f"{prefix}_{tag}"
            out_dir = ABL_DIR / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            missing = [c for c in CONDITIONS if not (out_dir / f"{c}.npy").exists()]
            if not missing:
                continue
            session = onnx_utils.make_session(ONNX_DIR / f"{variant}.onnx", C.NUM_THREADS)
            t0 = time.time()
            for cond in missing:
                images = test_x if cond == "clean" else np.load(CORRUPT_DIR / f"{cond}.npy", mmap_mode="r")
                np.save(out_dir / f"{cond}.npy", onnx_utils.ort_logits(session, images).astype(np.float16))
            print(f"[{variant}] {len(missing)} conditions in {time.time() - t0:.0f}s", flush=True)


def accuracies(pred_dir, test_y):
    acc = {}
    for cond in CONDITIONS:
        logits = np.load(pred_dir / f"{cond}.npy").astype(np.float32)
        acc[cond] = float((logits.argmax(1) == test_y).mean())
    return acc


def main():
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    build_models(d["train_x"][~is_val])
    run_inference(d["test_x"])

    rows = []
    for prefix, fp32_stem in STUDENTS.items():
        names = [("fp32", f"{prefix}_fp32" if prefix == "student" else "student_kd_fp32", PRED_DIR, fp32_stem + ".onnx"),
                 ("int8", f"{prefix}_int8", PRED_DIR, f"{prefix}_int8.onnx")]
        names += [(tag, f"{prefix}_{tag}", ABL_DIR, f"{prefix}_{tag}.onnx") for tag in VARIANT_NODES]
        for config, variant, pdir, onnx_name in names:
            acc = accuracies(pdir / variant, d["test_y"])
            row = {"model": prefix, "config": config, "variant": variant,
                   "size_MB": benchmark.file_size_mb(ONNX_DIR / onnx_name),
                   "clean_acc": acc["clean"],
                   "corr_acc_mean": float(np.mean([acc[c] for c in CONDITIONS[1:]]))}
            for corruption in C.CORRUPTIONS:
                row[f"{corruption}_mean"] = float(np.mean([acc[f"{corruption}_{s}"] for s in C.SEVERITIES]))
            for s in C.SEVERITIES:
                row[f"fog_sev{s}"] = acc[f"fog_{s}"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "fog_ablation.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
