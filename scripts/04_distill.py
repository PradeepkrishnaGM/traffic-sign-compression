"""Knowledge-distilled student, INT8 versions, and weather-calibrated INT8 variants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config as C
from src import data, models, train, onnx_utils, corruptions, benchmark

ONNX_DIR = C.CKPT_DIR / "onnx"


def train_kd_student(d, is_val):
    """Same epochs, optimizer, scheduler, seed and augmentation as the scratch student; only the loss differs."""
    tx, ty = d["train_x"][~is_val], d["train_y"][~is_val]
    vx, vy = d["train_x"][is_val], d["train_y"][is_val]
    teacher = models.make_teacher(pretrained=False)
    teacher.load_state_dict(torch.load(C.CKPT_DIR / "teacher_fp32.pt"))
    teacher.eval()
    torch.manual_seed(C.SEED)
    student = models.make_student()
    opt = torch.optim.Adam(student.parameters(), lr=C.STUDENT_LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.STUDENT_EPOCHS)
    history = train.fit(student, "student_kd", tx, ty, vx, vy, C.STUDENT_EPOCHS, opt, sched,
                        teacher=teacher)
    torch.save(student.state_dict(), C.CKPT_DIR / "student_kd_fp32.pt")
    return student, history


def main():
    torch.set_num_threads(C.NUM_THREADS)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    train_x = d["train_x"][~is_val]
    test_x, test_y = d["test_x"], d["test_y"]

    if (C.CKPT_DIR / "student_kd_fp32.pt").exists():
        student = models.make_student()
        student.load_state_dict(torch.load(C.CKPT_DIR / "student_kd_fp32.pt"))
        # training history (for the timing column) is stored inside the last checkpoint
        history = torch.load(C.CKPT_DIR / "student_kd_last.pt", weights_only=False)["history"]
    else:
        student, history = train_kd_student(d, is_val)
    student.eval()
    kd_acc = train.accuracy(student, test_x, test_y)
    print(f"KD student test acc {kd_acc:.4f}", flush=True)

    clean_calib = onnx_utils.calibration_images(train_x)
    weather_calib = corruptions.weather_calibration_set(train_x)
    np.save(C.CACHE_DIR / "wcal_calib.npy", weather_calib)

    onnx_utils.export_onnx(student, ONNX_DIR / "student_kd_fp32.onnx")
    # (model checkpoint, onnx fp32 file stem, name stem for the INT8 files)
    jobs = [("teacher", "teacher_fp32", "teacher"), ("student", "student_fp32", "student"),
            ("student_kd", "student_kd_fp32", "student_kd")]
    rows = []
    for _, fp32_stem, stem in jobs:
        fp32_path = ONNX_DIR / f"{fp32_stem}.onnx"
        for tag, calib in (("int8", clean_calib), ("int8_wcal", weather_calib)):
            out = ONNX_DIR / f"{stem}_{tag}.onnx"
            if not out.exists():
                onnx_utils.quantize_int8(fp32_path, out, calib)
            sess = onnx_utils.make_session(out, C.NUM_THREADS)
            pred = onnx_utils.ort_logits(sess, test_x).argmax(1)
            rows.append({"model": stem, "variant": tag, "size_MB": benchmark.file_size_mb(out),
                         "test_acc": float((pred == test_y).mean())})
            print(rows[-1], flush=True)
    rows.append({"model": "student_kd", "variant": "fp32", "size_MB": benchmark.file_size_mb(
        C.CKPT_DIR / "student_kd_fp32.pt"), "test_acc": kd_acc,
        "train_minutes": sum(h["seconds"] for h in history) / 60})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "distill_and_wcal.csv", index=False)


if __name__ == "__main__":
    main()
