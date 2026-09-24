"""Metrics, plots and macro-F1-aware probability calibration.

The competition metric is a single ``f1_score(y_true, y_pred, average="macro")``
over the 4 classes; everything here optimises exactly that quantity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import EMOTIONS, IDX_TO_EMOTION, setup_logging

LOG = logging.getLogger(__name__)

__all__ = [
    "macro_f1",
    "per_class_f1",
    "confusion",
    "plot_confusion",
    "classification_report",
    "temperature_scale",
    "fit_temperature",
    "apply_multipliers",
    "optimize_multipliers",
    "apply_calibration",
    "best_threshold_and_bias",
]

_DEFAULT_MULTIPLIERS = [1.0] * len(EMOTIONS)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Unweighted mean of per-class F1 (the competition metric)."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_f1(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None
) -> Dict[str, float]:
    """``{emotion: f1}`` for all 4 classes (missing classes -> 0.0)."""
    from sklearn.metrics import f1_score

    labels = list(range(len(EMOTIONS))) if labels is None else list(labels)
    scores = f1_score(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), labels=labels, average=None, zero_division=0)
    return {IDX_TO_EMOTION[i]: float(s) for i, s in zip(labels, scores)}


def confusion(y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Confusion matrix (rows = truth, cols = prediction)."""
    from sklearn.metrics import confusion_matrix

    labels = list(range(len(EMOTIONS))) if labels is None else list(labels)
    return confusion_matrix(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), labels=labels)


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], title: str = ""
) -> str:
    lines = [f"== {title} ==" if title else "=="]
    lines.append(f"macro F1: {macro_f1(y_true, y_pred):.4f}")
    per_class = per_class_f1(y_true, y_pred)
    for name, score in per_class.items():
        lines.append(f"  {name:<8s} F1 {score:.4f}")
    cm = confusion(y_true, y_pred)
    lines.append("confusion matrix (rows=true, cols=pred, order=%s):" % ",".join(EMOTIONS))
    for i, row in enumerate(cm):
        lines.append(f"  {EMOTIONS[i]:<8s} " + " ".join(f"{int(v):5d}" for v in row))
    return "\n".join(lines)


def plot_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: str | os.PathLike[str],
    title: str = "confusion matrix",
    normalize: bool = False,
) -> str:
    """Save a confusion-matrix PNG (matplotlib imported lazily)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion(y_true, y_pred)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    path = str(out_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=list(EMOTIONS),
        yticklabels=list(EMOTIONS),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nmacro F1 = {macro_f1(y_true, y_pred):.4f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOG.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def temperature_scale(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    """``softmax(log(p) / T)`` — T > 1 softens, T < 1 sharpens."""
    probs = np.asarray(probs, dtype=np.float64)
    t = max(float(temperature), eps)
    logits = np.log(np.clip(probs, eps, 1.0)) / t
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_temperature(
    probs: np.ndarray,
    y_true: Sequence[int],
    bounds: Tuple[float, float] = (0.25, 4.0),
) -> float:
    """Fit a single temperature by minimising OOF NLL (scipy, lazy import)."""
    from scipy.optimize import minimize_scalar

    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true).ravel().astype(int)
    if probs.shape[0] == 0:
        return 1.0

    def nll(log_t: float) -> float:
        p = temperature_scale(probs, float(np.exp(log_t))).astype(np.float64)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    t = float(np.exp(res.x))
    LOG.info("Fitted temperature T=%.4f (NLL %.4f)", t, float(res.fun))
    return t


def apply_multipliers(probs: np.ndarray, multipliers: Sequence[float], smoothing: float = 0.0) -> np.ndarray:
    """Multiply per-class probabilities, optionally smooth, renormalise."""
    p = np.asarray(probs, dtype=np.float64) * np.asarray(multipliers, dtype=np.float64).reshape(1, -1)
    if smoothing and smoothing > 0:
        n_classes = p.shape[1]
        p = p * (1.0 - float(smoothing)) + float(smoothing) / n_classes
    p = np.clip(p, 1e-12, None)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def optimize_multipliers(
    probs: np.ndarray,
    y_true: Sequence[int],
    lower: float = 0.6,
    upper: float = 1.4,
    steps: int = 16,
    rounds: int = 3,
    multipliers: Optional[Sequence[float]] = None,
    smoothing: float = 0.0,
) -> Tuple[List[float], float]:
    """Coordinate-ascent hill climb over per-class probability multipliers.

    Multipliers are searched on a ``steps``-point grid inside ``[lower, upper]``;
    each coordinate is optimised in turn for ``rounds`` passes while the others
    are frozen. Returns ``(best_multipliers, best_macro_f1)``.
    """
    y = np.asarray(y_true).ravel().astype(int)
    if probs.shape[0] == 0:
        return list(_DEFAULT_MULTIPLIERS), float("nan")

    grid = np.linspace(float(lower), float(upper), int(max(steps, 2)))
    best = list(multipliers) if multipliers else list(_DEFAULT_MULTIPLIERS)
    best_score = macro_f1(y, apply_multipliers(probs, best, smoothing).argmax(axis=1))

    for _ in range(int(max(rounds, 1))):
        improved = False
        for cls in range(len(EMOTIONS)):
            current = best[cls]
            for cand in grid:
                if abs(cand - current) < 1e-9:
                    continue
                trial = list(best)
                trial[cls] = float(cand)
                score = macro_f1(y, apply_multipliers(probs, trial, smoothing).argmax(axis=1))
                if score > best_score + 1e-9:
                    best, best_score, improved = trial, score, True
        if not improved:
            break
    return best, float(best_score)


def best_threshold_and_bias(
    probs: np.ndarray,
    y_true: Sequence[int],
    lower: float = 0.6,
    upper: float = 1.4,
    steps: int = 16,
    rounds: int = 3,
    fit_temperature_: bool = True,
    temperature_bounds: Tuple[float, float] = (0.5, 2.5),
) -> Dict[str, object]:
    """Full OOF calibration: temperature -> class multipliers -> smoothing floor.

    Returns a dict (JSON-serialisable) accepted by :func:`apply_calibration`::

        {"temperature": 1.05, "multipliers": [...], "smoothing": 0.0,
         "macro_f1": 0.912, "macro_f1_raw": 0.905, "n": 826}
    """
    probs = np.asarray(probs, dtype=np.float32)
    y = np.asarray(y_true).ravel().astype(int)
    raw = macro_f1(y, probs.argmax(axis=1)) if probs.size else float("nan")

    temperature = 1.0
    if fit_temperature_ and probs.size:
        try:
            temperature = fit_temperature(probs, y, bounds=temperature_bounds)
        except Exception as exc:  # pragma: no cover - scipy defensive
            LOG.warning("Temperature fitting failed (%s); using T=1.0", exc)
            temperature = 1.0
    scaled = temperature_scale(probs, temperature) if probs.size else probs

    multipliers, score = optimize_multipliers(
        scaled, y, lower=lower, upper=upper, steps=steps, rounds=rounds
    )
    # Optional "label smoothing floor": a small uniform floor can rescue
    # macro-F1 when one class is systematically under-predicted.
    smoothing = 0.0
    if probs.size:
        for cand in (0.0, 0.02, 0.05, 0.1):
            trial = macro_f1(y, apply_multipliers(scaled, multipliers, cand).argmax(axis=1))
            if trial > score + 1e-9:
                score, smoothing = float(trial), float(cand)

    result = {
        "temperature": float(temperature),
        "multipliers": [float(m) for m in multipliers],
        "smoothing": float(smoothing),
        "macro_f1": float(score),
        "macro_f1_raw": float(raw),
        "n": int(probs.shape[0]),
        "classes": list(EMOTIONS),
    }
    LOG.info(
        "Calibration: macro-F1 %.4f -> %.4f (T=%.3f, multipliers=%s, smoothing=%.2f)",
        raw, float(score), temperature, [round(m, 3) for m in multipliers], smoothing,
    )
    return result


def apply_calibration(probs: np.ndarray, calibration: Dict[str, object]) -> np.ndarray:
    """Apply a :func:`best_threshold_and_bias` dict to new probabilities."""
    scaled = temperature_scale(probs, float(calibration.get("temperature", 1.0) or 1.0))
    return apply_multipliers(
        scaled,
        calibration.get("multipliers") or _DEFAULT_MULTIPLIERS,
        float(calibration.get("smoothing", 0.0) or 0.0),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _probs_from_csv(path: str, true_col: str = "true") -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    prob_cols = [f"prob_{e}" for e in EMOTIONS]
    if not all(c in df.columns for c in prob_cols):
        raise ValueError(f"{path} must contain columns {prob_cols}")
    y = df[true_col].to_numpy().astype(int) if true_col in df.columns else np.zeros(len(df), dtype=int)
    return df[prob_cols].to_numpy(dtype=np.float32), y


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval", description="OOF metrics + calibration report.")
    parser.add_argument("--oof", required=True, help="CSV with filename,fold,true,prob_*.")
    parser.add_argument("--plot", default=None, help="Save confusion matrix PNG here.")
    parser.add_argument("--out-json", default=None, help="Save the calibration dict here.")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip the calibration search.")
    args = parser.parse_args(argv)

    setup_logging()
    probs, y_true = _probs_from_csv(args.oof)
    print(classification_report(y_true, probs.argmax(axis=1), title=args.oof))
    if args.plot:
        plot_confusion(y_true, probs.argmax(axis=1), args.plot, title=os.path.basename(args.oof))
    if not args.no_calibrate:
        calib = best_threshold_and_bias(probs, y_true)
        print(
            "calibrated macro F1: %.4f (raw %.4f)"
            % (float(calib["macro_f1"]), float(calib["macro_f1_raw"]))
        )
        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as fh:
                json.dump(calib, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
