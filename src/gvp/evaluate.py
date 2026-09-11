"""Metrics, plots and Markdown rendering used by every experiment."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABELS = ("plastic", "glass")


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                   y_score: Sequence[float] | None = None) -> Dict[str, float]:
    """Metric bundle. ``glass`` is the positive class; balanced accuracy is the headline."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "glass_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "glass_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "plastic_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out.update({
        "n": int(cm.sum()),
        "tn_plastic_ok": int(cm[0, 0]),
        "plastic_as_glass": int(cm[0, 1]),
        "glass_as_plastic": int(cm[1, 0]),
        "tp_glass_ok": int(cm[1, 1]),
        "glass_to_plastic_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        "plastic_to_glass_rate": float(cm[0, 1] / max(cm[0].sum(), 1)),
    })
    if y_score is not None and len(set(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int], metric: str = "balanced_accuracy",
                 n_boot: int = 400, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI. With ~100 test images the CI is wide -- report it."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    fn = {"balanced_accuracy": balanced_accuracy_score, "accuracy": accuracy_score,
          "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0)}[metric]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def plot_confusion(y_true: Sequence[int], y_pred: Sequence[int], path: str, title: str,
                   normalize: bool = False) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=140)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bars(labels: Sequence[str], values: Sequence[float], path: str, title: str,
              ylabel: str = "balanced accuracy", errs: Sequence[float] | None = None,
              highlight_best: bool = True) -> str:
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [float(values[i]) for i in order]
    errs = None if errs is None else [float(errs[i]) for i in order]
    fig, ax = plt.subplots(figsize=(max(5, 0.42 * len(labels) + 2), 3.6), dpi=140)
    colours = ["#2b6cb0"] * len(labels)
    if highlight_best and colours:
        colours[0] = "#c05621"
    ax.bar(range(len(labels)), values, color=colours,
           yerr=errs, capsize=3, error_kw={"elinewidth": 0.8})
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    lo = min(0.0, min(values) - 0.05)
    ax.set_ylim(max(0.0, lo - (0.05 if errs else 0.0)), 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(names: Sequence[str], importance: Sequence[float], path: str,
                            title: str, top_k: int = 24) -> str:
    order = np.argsort(np.abs(importance))[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(order) + 1.4), dpi=140)
    ax.barh(range(len(order))[::-1], np.asarray(importance)[order],
            color=["#2b6cb0" if importance[i] >= 0 else "#c05621" for i in order])
    ax.set_yticks(range(len(order))[::-1], [names[i] for i in order], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_matrix(row_labels: Sequence[str], col_labels: Sequence[str], values: np.ndarray,
                path: str, title: str, cbar_label: str = "balanced accuracy",
                vmin: float = 0.4, vmax: float = 1.0) -> str:
    """Heatmap for the robustness sweep (models x acquisition shifts)."""
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(0.75 * len(col_labels) + 3.2, 0.42 * len(row_labels) + 1.8),
                           dpi=140)
    im = ax.imshow(values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, label=cbar_label)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def md_table(rows: List[Dict[str, object]], columns: Sequence[str] | None = None,
             float_fmt: str = "{:.4f}") -> str:
    """Render a list of dicts as a Markdown table."""
    if not rows:
        return "_(no rows)_"
    columns = list(columns or rows[0].keys())
    def _fmt(v: object) -> str:
        if v is None:                 # a field the run did not record
            return "—"
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return float_fmt.format(v)
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


CAVEAT = (
    "> ⚠️ **Proxy-data caveat.** Numbers produced by the built-in synthetic renderer measure "
    "method behaviour on a *simulated* cue structure, not real-world glass/plastic accuracy. "
    "See `docs/03_datasets_and_benchmarks.md` for real-data benchmarks and "
    "`docs/09_evaluation_protocol.md` for the protocol to reproduce this on TrashNet/RealWaste."
)
