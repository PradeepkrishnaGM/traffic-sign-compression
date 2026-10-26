"""Metrics computed from saved logits: flip rate, rank correlation, calibration (ECE)."""
import numpy as np

from . import config as C


def softmax(logits):
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)      # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def flip_rate(pred, parent_pred):
    """Percentage of images where the predicted class differs from the FP32 parent's."""
    return float((pred != parent_pred).mean() * 100)


def _ranks(values):
    """Ranks 1..n (ties get the average rank)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        same = values == v
        ranks[same] = ranks[same].mean()
    return ranks


def spearman(a, b):
    """Spearman rank correlation = Pearson correlation of the ranks."""
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def reliability_bins(probs, labels, n_bins=C.ECE_BINS):
    """Per confidence bin: (count, mean confidence, accuracy). Used for ECE and the reliability diagram."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    # confidence exactly 1.0 belongs in the last bin
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = which == b
        n = int(mask.sum())
        out.append((n, float(conf[mask].mean()) if n else 0.0, float(correct[mask].mean()) if n else 0.0))
    return out


def ece(probs, labels, n_bins=C.ECE_BINS):
    """Expected calibration error: weighted average gap between confidence and accuracy per bin."""
    total = len(labels)
    return float(sum(n / total * abs(acc - conf)
                     for n, conf, acc in reliability_bins(probs, labels, n_bins)))


def calibration_summary(logits, labels):
    """ECE, mean confidence on wrong predictions, and over-confidence (mean confidence - accuracy)."""
    probs = softmax(logits)
    conf = probs.max(axis=1)
    wrong = probs.argmax(axis=1) != labels
    return {
        "ece": ece(probs, labels),
        "conf_wrong": float(conf[wrong].mean()) if wrong.any() else float("nan"),
        "overconfidence": float(conf.mean() - (~wrong).mean()),
        "acc": float((~wrong).mean()),
    }


def pareto_front(xs, ys):
    """Indices of points on the Pareto front (smaller x is better, larger y is better).

    A point is dominated if another point is at least as good on both axes and strictly
    better on one. So a point with the same x but lower y is dominated and is NOT on the
    front. Two points that are identical on both axes do not dominate each other.
    """
    front = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if j == i:
                continue
            as_good = xs[j] <= xs[i] and ys[j] >= ys[i]
            strictly_better = xs[j] < xs[i] or ys[j] > ys[i]
            if as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: (xs[i], -ys[i]))


def flip_rate_parent_correct(pred, parent_pred, labels):
    """Flip rate (%) counted only over images the FP32 parent classified correctly.

    This isolates 'the compressed model broke something that worked' from 'it changed
    an answer the parent already had wrong'.
    """
    ok = parent_pred == labels
    if ok.sum() == 0:
        return float("nan")
    return float((pred[ok] != parent_pred[ok]).mean() * 100)


def add_relative_metrics(table, test_y, pred_dir):
    """Add error ratios and parent-correct flip rates to the results table (from saved logits).

    err_ratio = (100 - acc) / (100 - parent_acc): 1.0 means as many errors as the parent,
    2.0 means twice as many. Corrupted values use the mean accuracy over the 30 sets.
    """
    t = table.set_index("variant")
    names = ["err_ratio_clean", "err_ratio_corr",
             "flip_rate_clean_parentcorrect", "flip_rate_corr_parentcorrect"]
    for n in names:
        t[n] = np.nan
    conds = [f"{c}_{s}" for c in C.CORRUPTIONS for s in C.SEVERITIES]
    for variant, row in t.iterrows():
        parent = row["parent"]
        if not isinstance(parent, str) or parent == "":
            continue
        p = t.loc[parent]
        t.loc[variant, "err_ratio_clean"] = (1 - row["clean_acc"]) / (1 - p["clean_acc"])
        t.loc[variant, "err_ratio_corr"] = (1 - row["corr_acc_mean"]) / (1 - p["corr_acc_mean"])
        load = lambda v, c: np.load(pred_dir / v / f"{c}.npy").argmax(1)
        t.loc[variant, "flip_rate_clean_parentcorrect"] = flip_rate_parent_correct(
            load(variant, "clean"), load(parent, "clean"), test_y)
        vals = [flip_rate_parent_correct(load(variant, c), load(parent, c), test_y) for c in conds]
        t.loc[variant, "flip_rate_corr_parentcorrect"] = float(np.mean(vals))
    return t.reset_index()
