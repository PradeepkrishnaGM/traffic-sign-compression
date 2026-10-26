"""Evaluate every variant on clean + 6 corruptions x 5 severities; derive all robustness metrics.

Step 1 builds the corrupted test sets (cache/corrupt/*.npy, uint8, built once).
Step 2 runs each variant and saves float16 logits to cache/preds/<variant>/<condition>.npy
        (resumable: existing files are skipped).
Step 3 computes every metric from the saved logits only, so no extra inference is needed.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import analysis, corruptions, data, evaluate, onnx_utils
from src.variants import ALL_VARIANTS, GROUP, VARIANTS

ONNX_DIR = C.CKPT_DIR / "onnx"
CORRUPT_DIR = C.CACHE_DIR / "corrupt"
PRED_DIR = C.CACHE_DIR / "preds"

CONDITIONS = ["clean"] + [f"{c}_{s}" for c in C.CORRUPTIONS for s in C.SEVERITIES]


def build_corrupted_sets(test_x):
    CORRUPT_DIR.mkdir(parents=True, exist_ok=True)
    for corruption in C.CORRUPTIONS:
        for severity in C.SEVERITIES:
            path = CORRUPT_DIR / f"{corruption}_{severity}.npy"
            if not path.exists():
                np.save(path, corruptions.corrupt(test_x, corruption, severity))


def corruption_grid_figure(test_x, test_y):
    """One sign x every corruption x every severity (also used to sanity-check difficulty)."""
    idx = int(np.where(test_y == 14)[0][3])      # a stop sign, easy for people to judge
    fig, axes = plt.subplots(len(C.CORRUPTIONS), 6, figsize=(9, 9))
    for r, corruption in enumerate(C.CORRUPTIONS):
        axes[r, 0].imshow(test_x[idx])
        axes[r, 0].set_ylabel(corruption, fontsize=9)
        axes[r, 0].set_title("clean" if r == 0 else "", fontsize=9)
        for s in C.SEVERITIES:
            img = np.load(CORRUPT_DIR / f"{corruption}_{s}.npy", mmap_mode="r")[idx]
            axes[r, s].imshow(img)
            axes[r, s].set_title(f"severity {s}" if r == 0 else "", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("GTSRB-Weather: one sign, six corruptions, five severities")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "corruption_grid.png", dpi=110)
    plt.close(fig)


def run_inference(test_x):
    for variant, (_, _, onnx_name, _) in ALL_VARIANTS.items():
        out_dir = PRED_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        missing = [c for c in CONDITIONS if not (out_dir / f"{c}.npy").exists()]
        if not missing:
            continue
        session = onnx_utils.make_session(ONNX_DIR / onnx_name, C.NUM_THREADS)
        t0 = time.time()
        for cond in missing:
            images = test_x if cond == "clean" else np.load(CORRUPT_DIR / f"{cond}.npy", mmap_mode="r")
            np.save(out_dir / f"{cond}.npy", onnx_utils.ort_logits(session, images).astype(np.float16))
        print(f"[{variant}] {len(missing)} conditions in {time.time() - t0:.0f}s", flush=True)


def load_logits(variant, cond):
    return np.load(PRED_DIR / variant / f"{cond}.npy").astype(np.float32)


def compute_metrics(test_y):
    long_rows, class_rows, calib_rows, summary = [], [], [], []
    for variant, (_, _, _, parent) in ALL_VARIANTS.items():
        clean_logits = load_logits(variant, "clean")
        clean_pred = clean_logits.argmax(1)
        clean_acc = float((clean_pred == test_y).mean())
        parent_clean = load_logits(parent, "clean").argmax(1) if parent else None

        accs, flips, group_acc = [], [], {"seen": [], "heldout": []}
        corr_logits_all, corr_labels_all, parent_pred_all, pred_all = [], [], [], []
        for corruption in C.CORRUPTIONS:
            for severity in C.SEVERITIES:
                cond = f"{corruption}_{severity}"
                logits = load_logits(variant, cond)
                pred = logits.argmax(1)
                acc = float((pred == test_y).mean())
                row = {"variant": variant, "corruption": corruption, "severity": severity, "acc": acc}
                if parent:
                    parent_pred = load_logits(parent, cond).argmax(1)
                    row["flip_rate"] = analysis.flip_rate(pred, parent_pred)
                    flips.append(row["flip_rate"])
                    parent_pred_all.append(parent_pred)
                row.update({k: v for k, v in analysis.calibration_summary(logits, test_y).items()
                            if k != "acc"})
                long_rows.append(row)
                accs.append(acc)
                group_acc["seen" if corruption in C.WCAL_CORRUPTIONS else "heldout"].append(acc)
                corr_logits_all.append(logits)
                pred_all.append(pred)
                corr_labels_all.append(test_y)

        pooled_logits = np.concatenate(corr_logits_all)
        pooled_labels = np.concatenate(corr_labels_all)
        cal_clean = analysis.calibration_summary(clean_logits, test_y)
        cal_corr = analysis.calibration_summary(pooled_logits, pooled_labels)
        calib_rows.append({"variant": variant, "condition": "clean", **cal_clean})
        calib_rows.append({"variant": variant, "condition": "corrupted", **cal_corr})

        pc = evaluate.per_class_accuracy(np.concatenate(pred_all), pooled_labels)
        for cls in range(C.NUM_CLASSES):
            class_rows.append({"variant": variant, "class": cls, "acc_corrupted": pc[cls],
                               "acc_clean": evaluate.per_class_accuracy(clean_pred, test_y)[cls]})
        summary.append({
            "variant": variant, "group": GROUP[variant], "parent": parent or "",
            "clean_acc": clean_acc, "corr_acc_mean": float(np.mean(accs)),
            "robustness_gap": clean_acc - float(np.mean(accs)),
            "corr_acc_seen": float(np.mean(group_acc["seen"])),
            "corr_acc_heldout": float(np.mean(group_acc["heldout"])),
            "flip_rate_clean": analysis.flip_rate(clean_pred, parent_clean) if parent else np.nan,
            "flip_rate_corr": float(np.mean(flips)) if parent else np.nan,
            "ece_clean": cal_clean["ece"], "ece_corr": cal_corr["ece"],
            "conf_wrong_clean": cal_clean["conf_wrong"], "conf_wrong_corr": cal_corr["conf_wrong"],
            "overconf_clean": cal_clean["overconfidence"], "overconf_corr": cal_corr["overconfidence"],
        })

    summary = pd.DataFrame(summary).set_index("variant")
    # drops relative to each variant's FP32 parent (percentage points)
    for col_src, col_dst in (("clean_acc", "clean_drop"), ("corr_acc_mean", "corr_drop")):
        summary[col_dst] = [
            (summary.loc[p, col_src] - summary.loc[v, col_src]) * 100 if p else np.nan
            for v, p in summary["parent"].items()]
    summary["corr_to_clean_drop_ratio"] = summary["corr_drop"] / summary["clean_drop"].where(
        summary["clean_drop"].abs() > 0.05)
    summary = summary.reset_index()

    pd.DataFrame(long_rows).to_csv(C.RESULTS_DIR / "robustness_long.csv", index=False)
    pd.DataFrame(class_rows).to_csv(C.RESULTS_DIR / "per_class_corr.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(C.RESULTS_DIR / "calibration.csv", index=False)
    summary.to_csv(C.RESULTS_DIR / "robustness_summary.csv", index=False)

    # does clean accuracy rank the variants like corruption accuracy does?
    rank_rows = []
    groups = {"all_variants": list(VARIANTS),
              "teacher_family": [v for v in VARIANTS if v.startswith("teacher")],
              "student_family": [v for v in VARIANTS if v.startswith("student")]}
    for name, members in groups.items():
        sub = summary[summary["variant"].isin(members)]
        rank_rows.append({"group": name, "n_variants": len(sub),
                          "spearman_clean_vs_corr": analysis.spearman(sub["clean_acc"], sub["corr_acc_mean"])})
    pd.DataFrame(rank_rows).to_csv(C.RESULTS_DIR / "rank_correlation.csv", index=False)

    # weather-aware calibration: seen vs held-out corruptions, for the fixed INT8 setting (main) and the
    # old ONNX Runtime default (int8_ort_default group)
    wrows = []
    for stem in ("teacher", "student", "student_kd"):
        for group, tag in (("main", "fp32"), ("main", "int8"), ("main", "int8_wcal"),
                           ("int8_ort_default", "int8_ortdefault"), ("int8_ort_default", "int8_wcal_ortdefault")):
            r = summary[summary["variant"] == f"{stem}_{tag}"].iloc[0]
            wrows.append({"group": group, "model": stem, "variant": tag, "clean_acc": r["clean_acc"],
                          "corr_acc_seen": r["corr_acc_seen"], "corr_acc_heldout": r["corr_acc_heldout"]})
    pd.DataFrame(wrows).to_csv(C.RESULTS_DIR / "wcal.csv", index=False)


def main():
    d = data.load_arrays()
    test_x, test_y = d["test_x"], d["test_y"]
    build_corrupted_sets(test_x)
    corruption_grid_figure(test_x, test_y)
    run_inference(test_x)
    compute_metrics(test_y)
    print("done", flush=True)


if __name__ == "__main__":
    main()
