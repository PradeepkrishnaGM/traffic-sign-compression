"""Training loop (resumable) and the knowledge-distillation loss."""
import json
import time
import torch
import torch.nn.functional as F

from . import config as C
from . import data


def kd_loss(student_logits, teacher_logits, labels, temperature=C.KD_TEMPERATURE, alpha=C.KD_ALPHA):
    """alpha * KL(teacher_T || student_T) * T^2 + (1 - alpha) * CE(student, labels).

    Dividing logits by T softens the distributions, so the student also sees how the
    teacher ranks the wrong classes. Multiplying by T^2 keeps the gradient size
    comparable to the CE term (the soft-target gradients shrink by 1/T^2).
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * temperature ** 2
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce


@torch.no_grad()
def predict_logits(model, images_uint8, batch_size=256):
    """Run a PyTorch model over uint8 NHWC images and return float32 logits (N, 43)."""
    model.eval()
    out = []
    for start in range(0, len(images_uint8), batch_size):
        x = data.to_tensor(images_uint8[start:start + batch_size])
        out.append(model(x))
    return torch.cat(out).numpy()


def accuracy(model, images, labels):
    logits = predict_logits(model, images)
    return float((logits.argmax(1) == labels).mean())


def fit(model, name, train_x, train_y, val_x, val_y, epochs, optimizer, scheduler=None,
        teacher=None, seed=C.SEED):
    """Train for `epochs`, saving a checkpoint after each epoch so a restart resumes.

    The weights with the best *validation* accuracy are restored at the end
    (the test set is never used for any choice).
    Returns the history list (one dict per epoch).
    """
    C.CKPT_DIR.mkdir(exist_ok=True)
    C.LOG_DIR.mkdir(exist_ok=True)
    last_path = C.CKPT_DIR / f"{name}_last.pt"
    best_path = C.CKPT_DIR / f"{name}_best.pt"
    history, start_epoch, best_val = [], 0, -1.0

    if last_path.exists():
        state = torch.load(last_path, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, best_val = state["history"], state["epoch"], state["best_val"]
        print(f"[{name}] resuming from epoch {start_epoch}", flush=True)

    if teacher is not None:
        teacher.eval()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total_loss, n_seen = 0.0, 0
        # a different shuffle/augmentation seed each epoch, but fixed for a given epoch
        for x, y in data.batches(train_x, train_y, C.BATCH_SIZE, train=True, seed=seed * 1000 + epoch):
            optimizer.zero_grad()
            logits = model(x)
            if teacher is None:
                loss = F.cross_entropy(logits, y)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss = kd_loss(logits, teacher_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_seen += len(y)
        if scheduler is not None:
            scheduler.step()
        val_acc = accuracy(model, val_x, val_y)
        row = {"epoch": epoch + 1, "train_loss": total_loss / n_seen, "val_acc": val_acc,
               "seconds": time.time() - t0}
        history.append(row)
        print(f"[{name}] {json.dumps(row)}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history, "epoch": epoch + 1, "best_val": best_val}, last_path)

    model.load_state_dict(torch.load(best_path))
    return history
