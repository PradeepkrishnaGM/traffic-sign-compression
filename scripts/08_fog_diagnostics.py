"""Exploratory follow-up, part 2: latency of the ablation models, per-layer INT8 error, figure.

Run only on an idle machine (latency is measured first, before anything else heavy).
Writes results/layer_error.csv, adds latency columns to results/fog_ablation.csv and
saves results/figures/fog_ablation.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from onnx import numpy_helper

from src import config as C
from src import benchmark, data, onnx_utils

ONNX_DIR = C.CKPT_DIR / "onnx"
CORRUPT_DIR = C.CACHE_DIR / "corrupt"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else C.CACHE_DIR / "tmp_diag"

# the six conv blocks (output after ReLU), global pooling and the logits
LAYERS = [("conv1_relu", "/features/features.0/features.0.2/Relu_output_0"),
          ("conv2_relu", "/features/features.0/features.0.5/Relu_output_0"),
          ("conv3_relu", "/features/features.1/features.1.2/Relu_output_0"),
          ("conv4_relu", "/features/features.1/features.1.5/Relu_output_0"),
          ("conv5_relu", "/features/features.2/features.2.2/Relu_output_0"),
          ("conv6_relu", "/features/features.2/features.2.5/Relu_output_0"),
          ("global_pool", "/pool/GlobalAveragePool_output_0"),
          ("logits", "logits")]


def session_with_outputs(path, tensor_names, tag):
    """ONNX Runtime session that also returns the given intermediate tensors.

    All graph optimisations are switched off so that no fusion hides an intermediate tensor
    and both models are run node by node.
    """
    model = onnx.load(str(path))
    existing = {o.name for o in model.graph.output}
    for name in tensor_names:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"{tag}.onnx"
    onnx.save(model, str(tmp))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = C.NUM_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(tmp), opts, providers=["CPUExecutionProvider"])


def quantizer_scale(int8_path, float_tensor):
    """Step size of the INT8 quantizer that follows a float tensor (None if there is none)."""
    model = onnx.load(str(int8_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] == float_tensor:
            return float(inits[node.input[1]])
    return None


def layer_errors(model_name, fp32_stem, int8_stem, images_by_condition):
    fp32_path = ONNX_DIR / f"{fp32_stem}_pre.onnx"       # the graph that was quantized
    int8_path = ONNX_DIR / f"{int8_stem}.onnx"
    names = [t for _, t in LAYERS]
    int8_names = [t if t == "logits" else t + "_DequantizeLinear_Output" for t in names]
    fp = session_with_outputs(fp32_path, names, f"{fp32_stem}_diag")
    q = session_with_outputs(int8_path, int8_names, f"{int8_stem}_diag")
    rows = []
    for condition, images in images_by_condition.items():
        x = data.to_tensor(images).numpy()
        out_fp = fp.run(names, {"input": x})
        out_q = q.run(int8_names, {"input": x})
        rows.append({"model": model_name, "layer": "input", "condition": condition,
                     "mean_abs_err": np.nan, "mean_abs_fp32": float(np.abs(x).mean()),
                     "contrast_std": float(x.reshape(len(x), -1).std(axis=1).mean()),
                     "err_over_contrast": np.nan, "quant_step": np.nan})
        for (label, tensor), a, b in zip(LAYERS, out_fp, out_q):
            err = float(np.abs(a - b).mean())
            contrast = float(a.reshape(len(a), -1).std(axis=1).mean())   # spread of the FP32 activation inside one image
            rows.append({"model": model_name, "layer": label, "condition": condition,
                         "mean_abs_err": err, "mean_abs_fp32": float(np.abs(a).mean()),
                         "contrast_std": contrast, "err_over_contrast": err / contrast,
                         "quant_step": quantizer_scale(int8_path, tensor)})
    return rows


def latency_table():
    rows = []
    for prefix in ("student", "student_kd"):
        for config, stem in (("fp32", f"{prefix}_fp32"), ("int8", f"{prefix}_int8"),
                             ("int8_fp32first", f"{prefix}_int8_fp32first"),
                             ("int8_fp32fc", f"{prefix}_int8_fp32fc"),
                             ("int8_fp32both", f"{prefix}_int8_fp32both")):
            sample = np.load(C.CACHE_DIR / "test_x.npy", mmap_mode="r")[:1]
            med, p90 = benchmark.ort_latency(ONNX_DIR / f"{stem}.onnx", np.array(sample))
            rows.append({"variant": stem, "latency_ms_cpu": med, "latency_p90_ms": p90})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def figure(table):
    colors = {"fp32": "#2a78d6", "int8": "#eb6834", "int8_fp32first": "#1baf7a",
              "int8_fp32fc": "#eda100", "int8_fp32both": "#6250d6"}
    markers = {"fp32": "o", "int8": "s", "int8_fp32first": "D", "int8_fp32fc": "^", "int8_fp32both": "v"}
    labels = {"fp32": "FP32", "int8": "INT8 (plain)", "int8_fp32first": "INT8, first conv in FP32",
              "int8_fp32fc": "INT8, final linear in FP32", "int8_fp32both": "INT8, both in FP32"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (prefix, title) in zip(axes, (("student", "Scratch student"), ("student_kd", "Distilled student"))):
        ax.set_facecolor("white")
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # several configurations give (almost) identical curves, so the widths and dashes differ:
        # thick lines are drawn first, thin dashed ones on top of them
        style = {"fp32": (7, "-", 1), "int8": (7, "-", 1), "int8_fp32first": (3.5, "-", 2),
                 "int8_fp32fc": (2.2, (0, (3, 2)), 3), "int8_fp32both": (1.6, "-", 4)}
        for config in labels:
            r = table[(table["model"] == prefix) & (table["config"] == config)].iloc[0]
            ys = [r[f"fog_sev{s}"] * 100 for s in C.SEVERITIES]
            width, dash, z = style[config]
            ax.plot(C.SEVERITIES, ys, color=colors[config], marker=markers[config], markersize=5,
                    markeredgecolor="white", linewidth=width, linestyle=dash, zorder=z, label=labels[config])
        ax.text(0.03, 0.05, "FP32 = first conv FP32 = both FP32 (overlap)\nplain INT8 = final linear FP32 (overlap)",
                transform=ax.transAxes, fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("fog severity")
        ax.set_xticks(C.SEVERITIES)
    axes[0].set_ylabel("test accuracy under fog (%)")
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Exploratory, single seed: which INT8 layers fail in fog?", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(C.FIG_DIR / "fog_ablation.png", dpi=120, facecolor="white")
    plt.close(fig)


def main():
    table = pd.read_csv(C.RESULTS_DIR / "fog_ablation.csv")
    lat = latency_table()                                  # first: nothing else heavy has run yet
    table = table.drop(columns=[c for c in ("latency_ms_cpu", "latency_p90_ms") if c in table]).merge(lat, on="variant")
    table.to_csv(C.RESULTS_DIR / "fog_ablation.csv", index=False)

    d = data.load_arrays()
    rng = np.random.RandomState(C.SEED)
    idx = np.sort(rng.choice(len(d["test_y"]), C.LAYER_ERROR_IMAGES, replace=False))
    fog5 = np.load(CORRUPT_DIR / "fog_5.npy", mmap_mode="r")
    images = {"clean": d["test_x"][idx], "fog_5": np.array(fog5[idx])}   # the same 500 images, clean and with fog
    rows = layer_errors("student", "student_fp32", "student_int8", images)
    rows += layer_errors("student_kd", "student_kd_fp32", "student_kd_int8", images)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "layer_error.csv", index=False)
    figure(table)
    print("done", flush=True)


if __name__ == "__main__":
    main()
