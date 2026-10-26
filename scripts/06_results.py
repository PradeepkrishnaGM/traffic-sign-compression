"""Latency benchmark (run with nothing else running), results.csv and all figures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src import config as C
from src import analysis, benchmark, data, models
from src.variants import ALL_VARIANTS, GROUP, TORCH_CKPT, VARIANTS

ONNX_DIR = C.CKPT_DIR / "onnx"
PRED_DIR = C.CACHE_DIR / "preds"

# colour = model family (fixed order), marker = compression method
FAMILY_COLOR = {"teacher": "#2a78d6", "student": "#eb6834", "student_kd": "#1baf7a"}
METHOD_MARKER = {"fp32": "o", "fp32_scratch": "o", "fp32_kd": "o", "int8_static": "s", "kd_int8": "s",
                 "int8_static_wcal": "D", "kd_int8_static_wcal": "D",
                 "prune_50": "^", "prune_70": "^", "prune_90": "^"}
INK, GRID = "#0b0b0b", "#e4e3df"


def family_of(variant):
    if variant.startswith("student_kd"):
        return "student_kd"
    return "teacher" if variant.startswith("teacher") else "student"


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def short(variant):
    return variant.replace("student_kd", "kd").replace("teacher", "T").replace("student", "S")


# ---------------------------------------------------------------- latency + table
def run_latency(sample):
    rows = []
    for variant, (family, method, onnx_name, _) in ALL_VARIANTS.items():
        med, p90 = benchmark.ort_latency(ONNX_DIR / onnx_name, sample)
        row = {"variant": variant, "latency_ms_cpu": med, "latency_p90_ms": p90}
        if variant in TORCH_CKPT:
            kind, ckpt = TORCH_CKPT[variant]
            net = models.make_model(kind)
            net.load_state_dict(torch.load(C.CKPT_DIR / ckpt))
            row["pytorch_fp32_latency_ms"], _ = benchmark.torch_latency(net, sample)
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "latency.csv", index=False)
    return out


def count_params(variant):
    """(total params, non-zero conv/linear weights + other params) from the PyTorch checkpoint."""
    source = variant
    if variant not in TORCH_CKPT:        # INT8 variants have the same number of weights as their FP32 source
        source = variant.replace("_ortdefault", "").replace("_int8_wcal", "_fp32").replace("_int8", "_fp32")
        if variant.startswith("student_kd") and source == "student_kd_fp32":
            pass
    kind, ckpt = TORCH_CKPT[source]
    state = torch.load(C.CKPT_DIR / ckpt)
    weights = [v for k, v in state.items() if v.dtype.is_floating_point and
               not k.endswith(("running_mean", "running_var"))]
    total = sum(v.numel() for v in weights)
    nonzero = sum(int((v != 0).sum()) for v in weights)
    return total, nonzero


def build_results_table(latency, test_y):
    summary = pd.read_csv(C.RESULTS_DIR / "robustness_summary.csv").set_index("variant")
    prune = pd.read_csv(C.RESULTS_DIR / "pruning.csv").set_index("amount")
    rows = []
    for variant, (family, method, onnx_name, parent) in ALL_VARIANTS.items():
        path = ONNX_DIR / onnx_name
        total, nonzero = count_params(variant)
        s = summary.loc[variant]
        lat = latency.set_index("variant").loc[variant]
        note = ""
        if variant.endswith("_ortdefault"):
            note = ("old INT8 setting (full 8-bit signed weights): saturates in ONNX Runtime's U8S8 kernel on "
                    "this AVX2 CPU without VNNI; kept for comparison")
        elif "int8" in variant:
            note = "INT8 with reduce_range=True (7-bit weights)"
        if method.startswith("prune"):
            amount = int(method.split("_")[1]) / 100
            note = (f"unstructured L1, fine-tuned; test acc before fine-tune "
                    f"{prune.loc[amount, 'test_acc_before_ft']:.4f}; dense latency unchanged")
        rows.append({
            "model": family, "method": method, "variant": variant, "group": GROUP[variant], "parent": parent or "",
            "params_M": total / 1e6, "nonzero_params_M": nonzero / 1e6,
            "size_MB": benchmark.file_size_mb(path), "size_gzip_MB": benchmark.gzip_size_mb(path),
            "latency_ms_cpu": lat["latency_ms_cpu"], "latency_p90_ms": lat["latency_p90_ms"],
            "pytorch_fp32_latency_ms": lat.get("pytorch_fp32_latency_ms", np.nan),
            "clean_acc": s["clean_acc"], "corr_acc_mean": s["corr_acc_mean"],
            "robustness_gap": s["robustness_gap"], "clean_drop": s["clean_drop"], "corr_drop": s["corr_drop"],
            "flip_rate_clean": s["flip_rate_clean"], "flip_rate_corr": s["flip_rate_corr"],
            "ece_clean": s["ece_clean"], "ece_corr": s["ece_corr"], "notes": note})
    table = pd.DataFrame(rows)
    table = analysis.add_relative_metrics(table, test_y, PRED_DIR)
    table.to_csv(C.RESULTS_DIR / "results.csv", index=False)
    return table


# ---------------------------------------------------------------- figures
def pareto_figure(table, x_col, x_label, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, y_col, title in zip(axes, ("clean_acc", "corr_acc_mean"),
                                ("Clean test accuracy", "Mean accuracy under corruptions")):
        style_axes(ax)
        for _, r in table.iterrows():
            ax.scatter(r[x_col], r[y_col] * 100, s=70, marker=METHOD_MARKER[r["method"]],
                       color=FAMILY_COLOR[family_of(r["variant"])], edgecolor="white", linewidth=1.2, zorder=3)
        # Pareto front: points no other point beats on both smaller x and higher accuracy
        # (ties handled in analysis.pareto_front: same x and lower accuracy is dominated)
        pts = table[[x_col, y_col]].values
        keep = analysis.pareto_front(list(pts[:, 0]), list(pts[:, 1]))
        front = pts[keep]
        ax.step(front[:, 0], front[:, 1] * 100, where="post", color=INK, linewidth=1.0, alpha=0.6, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(title, fontsize=11)
    handles = [Line2D([], [], marker="o", color=c, linestyle="", markersize=8, label=f)
               for f, c in FAMILY_COLOR.items()]
    handles += [Line2D([], [], marker=m, color="#52514e", linestyle="", markersize=8, label=l)
                for m, l in (("o", "FP32"), ("s", "INT8"), ("D", "INT8 weather-calibrated"), ("^", "pruned"))]
    handles.append(Line2D([], [], color=INK, alpha=0.6, label="Pareto front"))
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(C.FIG_DIR / filename, dpi=120, facecolor="white")
    plt.close(fig)


def heatmap_figure():
    long = pd.read_csv(C.RESULTS_DIR / "robustness_long.csv")
    grid = long.groupby(["variant", "corruption"])["acc"].mean().unstack()[list(C.CORRUPTIONS)]
    grid = grid.loc[list(VARIANTS)] * 100
    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(C.CORRUPTIONS)), C.CORRUPTIONS, rotation=30, ha="right")
    ax.set_yticks(range(len(grid)), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else INK)
    ax.set_title("Mean accuracy (%) over severities 1-5")
    fig.colorbar(im, ax=ax, label="accuracy (%)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "heatmap_variant_corruption.png", dpi=120, facecolor="white")
    plt.close(fig)


def drop_bars_figure(table):
    sub = table[table["parent"] != ""]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    style_axes(ax)
    ax.bar(x - 0.2, sub["clean_drop"], 0.38, color="#2a78d6", label="drop on clean data")
    ax.bar(x + 0.2, sub["corr_drop"], 0.38, color="#eb6834", label="drop under corruptions (mean)")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, [v for v in sub["variant"]], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("accuracy drop vs. FP32 parent (percentage points)")
    ax.set_title("Does compression cost more under corruption than on clean data?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "clean_vs_corrupted_drop.png", dpi=120, facecolor="white")
    plt.close(fig)


def flip_figure(table):
    sub = table[table["parent"] != ""]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    style_axes(ax)
    ax.bar(x - 0.2, sub["flip_rate_clean"], 0.38, color="#2a78d6", label="clean")
    ax.bar(x + 0.2, sub["flip_rate_corr"], 0.38, color="#eb6834", label="corrupted (mean)")
    ax.set_xticks(x, list(sub["variant"]), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("prediction flip rate vs. FP32 parent (%)")
    ax.set_title("How often does the compressed model change the parent's decision?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "flip_rate.png", dpi=120, facecolor="white")
    plt.close(fig)


def int8_fix_figure(table):
    """Old ONNX Runtime default INT8 vs the reduce_range INT8, per model; the black tick is the FP32 parent."""
    pairs = [("teacher_int8", "teacher INT8"), ("teacher_int8_wcal", "teacher INT8 wcal"),
             ("student_int8", "student INT8"), ("student_int8_wcal", "student INT8 wcal"),
             ("student_kd_int8", "KD student INT8"), ("student_kd_int8_wcal", "KD student INT8 wcal")]
    t = table.set_index("variant")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    x = np.arange(len(pairs))
    for ax, col, title in zip(axes, ("clean_acc", "corr_acc_mean"), ("Clean accuracy (%)", "Mean accuracy under corruptions (%)")):
        style_axes(ax)
        old = [t.loc[v + "_ortdefault", col] * 100 for v, _ in pairs]
        new = [t.loc[v, col] * 100 for v, _ in pairs]
        parent = [t.loc[t.loc[v, "parent"], col] * 100 for v, _ in pairs]
        ax.bar(x - 0.2, old, 0.38, color="#eb6834", label="old default (saturating)")
        ax.bar(x + 0.2, new, 0.38, color="#2a78d6", label="reduce_range=True")
        ax.scatter(x, parent, marker="_", s=900, color=INK, linewidths=2, zorder=5, label="FP32 parent")
        ax.set_xticks(x, [lab for _, lab in pairs], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(min(old + new) - 3, max(old + new + parent) + 1.5)
        ax.set_title(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("The INT8 accuracy loss was an ONNX Runtime kernel effect on this CPU", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(C.FIG_DIR / "int8_fix.png", dpi=120, facecolor="white")
    plt.close(fig)


def severity_figure():
    long = pd.read_csv(C.RESULTS_DIR / "robustness_long.csv")
    shown = ["teacher_fp32", "teacher_int8", "teacher_prune90", "student_fp32", "student_kd_fp32",
             "student_kd_int8"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, corruption in zip(axes.ravel(), C.CORRUPTIONS):
        style_axes(ax)
        for variant, color in zip(shown, colors):
            d = long[(long["variant"] == variant) & (long["corruption"] == corruption)].sort_values("severity")
            ax.plot(d["severity"], d["acc"] * 100, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor="white", label=variant)
        ax.set_title(corruption, fontsize=10)
        ax.set_xticks(C.SEVERITIES)
    for ax in axes[1]:
        ax.set_xlabel("severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(C.FIG_DIR / "severity_curves.png", dpi=120, facecolor="white")
    plt.close(fig)


def reliability_figure(test_y):
    shown = [("teacher_fp32", "#2a78d6"), ("student_kd_int8", "#eb6834")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for r, (variant, color) in enumerate(shown):
        clean = np.load(PRED_DIR / variant / "clean.npy").astype(np.float32)
        pooled = np.concatenate([np.load(PRED_DIR / variant / f"{c}_{s}.npy").astype(np.float32)
                                 for c in C.CORRUPTIONS for s in C.SEVERITIES])
        labels_pooled = np.tile(test_y, len(C.CORRUPTIONS) * len(C.SEVERITIES))
        for c, (name, logits, labels) in enumerate((("clean", clean, test_y),
                                                    ("corrupted (all 30 sets)", pooled, labels_pooled))):
            ax = axes[r, c]
            style_axes(ax)
            bins = analysis.reliability_bins(analysis.softmax(logits), labels)
            centres = (np.arange(C.ECE_BINS) + 0.5) / C.ECE_BINS
            counts = np.array([b[0] for b in bins])
            accs = np.array([b[2] for b in bins])
            ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1)
            ax.bar(centres[counts > 0], accs[counts > 0], width=0.9 / C.ECE_BINS, color=color)
            ece = analysis.ece(analysis.softmax(logits), labels)
            ax.set_title(f"{variant}, {name}\nECE = {ece:.3f}", fontsize=10)
            if r == 1:
                ax.set_xlabel("confidence")
            if c == 0:
                ax.set_ylabel("accuracy in bin")
    fig.suptitle("Reliability diagrams (bars below the dashed line = over-confident)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "reliability_diagram.png", dpi=120, facecolor="white")
    plt.close(fig)


def main():
    d = data.load_arrays()
    latency = run_latency(d["test_x"])          # nothing else may be running now
    table = build_results_table(latency, d["test_y"])
    main_table = table[table["group"] == "main"].reset_index(drop=True)   # figures show the fixed INT8 group
    pareto_figure(main_table, "size_MB", "model file size (MB, ONNX, log scale)", "pareto_size.png")
    pareto_figure(main_table, "latency_ms_cpu", "CPU latency (ms, batch 1, 1 thread, log scale)", "pareto_latency.png")
    heatmap_figure()
    drop_bars_figure(main_table)
    flip_figure(main_table)
    int8_fix_figure(table)
    severity_figure()
    reliability_figure(d["test_y"])
    print("done", flush=True)


if __name__ == "__main__":
    main()
