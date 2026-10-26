"""Train teacher (ResNet-18) and student (small CNN) from scratch. Resumable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config as C
from src import data, models, train, evaluate, benchmark


def main():
    torch.set_num_threads(C.NUM_THREADS)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    C.RESULTS_DIR.mkdir(exist_ok=True)

    d = data.load_arrays()
    is_val = data.split_by_track(d["train_y"], d["train_track"])
    tx, ty = d["train_x"][~is_val], d["train_y"][~is_val]
    vx, vy = d["train_x"][is_val], d["train_y"][is_val]

    rows = []
    for name in ("teacher", "student"):
        torch.manual_seed(C.SEED)
        model = models.make_model(name)
        if name == "teacher":
            opt = torch.optim.SGD(model.parameters(), lr=C.TEACHER_LR, momentum=0.9,
                                  weight_decay=C.WEIGHT_DECAY)
            epochs = C.TEACHER_EPOCHS
        else:
            opt = torch.optim.Adam(model.parameters(), lr=C.STUDENT_LR, weight_decay=C.WEIGHT_DECAY)
            epochs = C.STUDENT_EPOCHS
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        history = train.fit(model, name, tx, ty, vx, vy, epochs, opt, sched)

        # final weights = best-validation epoch; the test set is touched only here
        logits = train.predict_logits(model, d["test_x"])
        pred = logits.argmax(1)
        test_acc = float((pred == d["test_y"]).mean())
        val_acc = train.accuracy(model, vx, vy)
        pc = evaluate.per_class_accuracy(pred, d["test_y"])
        pd.DataFrame({"class": range(C.NUM_CLASSES), "acc": pc}).to_csv(
            C.RESULTS_DIR / f"per_class_clean_{name}_fp32.csv", index=False)

        torch.save(model.state_dict(), C.CKPT_DIR / f"{name}_fp32.pt")
        size = benchmark.file_size_mb(C.CKPT_DIR / f"{name}_fp32.pt")
        rows.append({"model": name, "method": "fp32" if name == "teacher" else "fp32_scratch",
                     "params_M": models.count_params(model) / 1e6, "size_MB": size,
                     "val_acc": val_acc, "test_acc": test_acc,
                     "epochs": len(history),
                     "train_minutes": sum(h["seconds"] for h in history) / 60})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "baselines.csv", index=False)


if __name__ == "__main__":
    main()
