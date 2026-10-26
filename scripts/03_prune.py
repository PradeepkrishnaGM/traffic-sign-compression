"""Global unstructured L1 pruning of the teacher at 50/70/90%, with fine-tuning. Resumable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import prune

from src import config as C
from src import data, models, train, onnx_utils, benchmark

ONNX_DIR = C.CKPT_DIR / "onnx"


def prunable_parameters(model):
    """Weights of every conv and linear layer (BatchNorm and biases are tiny and left alone)."""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def measured_sparsity(model):
    """Fraction of exactly-zero weights among all conv/linear weights."""
    zeros, total = 0, 0
    for module, _ in prunable_parameters(model):
        zeros += int((module.weight == 0).sum())
        total += module.weight.numel()
    return zeros / total


def main():
    torch.set_num_threads(C.NUM_THREADS)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    tx, ty = d["train_x"][~is_val], d["train_y"][~is_val]
    vx, vy = d["train_x"][is_val], d["train_y"][is_val]
    done = C.RESULTS_DIR / "pruning.csv"
    rows = pd.read_csv(done).to_dict("records") if done.exists() else []
    finished = {r["amount"] for r in rows}

    for amount in C.PRUNE_AMOUNTS:
        if amount in finished:
            continue
        name = f"teacher_prune{int(amount * 100)}"
        torch.manual_seed(C.SEED)
        model = models.make_teacher(pretrained=False)
        model.load_state_dict(torch.load(C.CKPT_DIR / "teacher_fp32.pt"))

        # global = one shared threshold over all layers, so layers with small weights lose more
        prune.global_unstructured(prunable_parameters(model), pruning_method=prune.L1Unstructured,
                                  amount=amount)
        sparsity = measured_sparsity(model)
        acc_before = train.accuracy(model, d["test_x"], d["test_y"])
        print(f"[{name}] sparsity {sparsity:.4f}, test acc before fine-tune {acc_before:.4f}", flush=True)

        # fine-tune with the masks still attached, so pruned weights stay at zero
        opt = torch.optim.SGD(model.parameters(), lr=C.PRUNE_FT_LR, momentum=0.9,
                              weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.PRUNE_FT_EPOCHS)
        history = train.fit(model, name, tx, ty, vx, vy, C.PRUNE_FT_EPOCHS, opt, sched)

        for module, param in prunable_parameters(model):
            prune.remove(module, param)     # bake the mask into the weights (plain dense tensor)
        sparsity_after = measured_sparsity(model)
        logits = train.predict_logits(model, d["test_x"])
        acc_after = float((logits.argmax(1) == d["test_y"]).mean())

        pt_path = C.CKPT_DIR / f"{name}.pt"
        torch.save(model.state_dict(), pt_path)
        onnx_path = ONNX_DIR / f"{name}_fp32.onnx"
        onnx_utils.export_onnx(model, onnx_path)
        rows.append({
            "amount": amount, "model": name,
            "sparsity_after_prune": sparsity, "sparsity_after_finetune": sparsity_after,
            "test_acc_before_ft": acc_before, "test_acc_after_ft": acc_after,
            "dense_size_MB": benchmark.file_size_mb(pt_path),
            "gzip_size_MB": benchmark.gzip_size_mb(pt_path),
            "onnx_size_MB": benchmark.file_size_mb(onnx_path),
            "onnx_gzip_MB": benchmark.gzip_size_mb(onnx_path),
            "ft_minutes": sum(h["seconds"] for h in history) / 60,
        })
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(done, index=False)


if __name__ == "__main__":
    main()
