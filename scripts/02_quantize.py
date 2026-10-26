"""Export teacher and student to ONNX, quantize to INT8, check accuracy against PyTorch."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config as C
from src import data, models, train, onnx_utils, benchmark

ONNX_DIR = C.CKPT_DIR / "onnx"


def main():
    torch.set_num_threads(C.NUM_THREADS)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    train_x = d["train_x"][~is_val]            # calibration images come from the training split only
    calib = onnx_utils.calibration_images(train_x)
    test_x, test_y = d["test_x"], d["test_y"]

    rows = []
    for name in ("teacher", "student"):
        model = models.make_model(name)
        model.load_state_dict(torch.load(C.CKPT_DIR / f"{name}_fp32.pt"))
        model.eval()
        torch_logits = train.predict_logits(model, test_x)
        torch_acc = float((torch_logits.argmax(1) == test_y).mean())

        fp32_path = ONNX_DIR / f"{name}_fp32.onnx"
        int8_path = ONNX_DIR / f"{name}_int8.onnx"
        onnx_utils.export_onnx(model, fp32_path)
        t0 = time.time()
        onnx_utils.quantize_int8(fp32_path, int8_path, calib)
        print(f"[{name}] quantization took {time.time() - t0:.0f}s", flush=True)

        for tag, path in (("onnx_fp32", fp32_path), ("int8_static", int8_path)):
            sess = onnx_utils.make_session(path, C.NUM_THREADS)
            logits = onnx_utils.ort_logits(sess, test_x)
            pred = logits.argmax(1)
            rows.append({
                "model": name, "variant": tag,
                "size_MB": benchmark.file_size_mb(path),
                "test_acc": float((pred == test_y).mean()),
                "pytorch_fp32_acc": torch_acc,
                # how many test images get a different class than the PyTorch FP32 model
                "disagree_with_pytorch_pct": float((pred != torch_logits.argmax(1)).mean() * 100),
                "max_abs_logit_diff": float(np.abs(logits - torch_logits).max()),
            })
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "quantization.csv", index=False)


if __name__ == "__main__":
    main()
