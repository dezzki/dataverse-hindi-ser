"""Stratified K-fold generation with on-disk persistence.

Fold assignments are keyed by audio file name so that repeated runs, different
models and ensemble members all share the same validation split.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

__all__ = [
    "make_stratified_folds",
    "load_or_create_folds",
    "fold_split",
    "save_folds",
    "load_folds",
]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def make_stratified_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    label_col: str = "emotion",
) -> pd.DataFrame:
    """Return a copy of ``df`` with an integer ``fold`` column.

    Uses ``sklearn.model_selection.StratifiedKFold`` on ``label_col`` so each
    fold preserves the class distribution.
    """
    from sklearn.model_selection import StratifiedKFold

    if label_col not in df.columns:
        raise ValueError(f"Column {label_col!r} required for stratification")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    min_class = int(df[label_col].value_counts().min())
    if n_folds > min_class:
        raise ValueError(
            f"n_folds={n_folds} exceeds the smallest class count ({min_class}); "
            "reduce num_folds or merge classes."
        )

    out = df.copy().reset_index(drop=True)
    folds = np.full(len(out), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    for fold, (_, val_idx) in enumerate(skf.split(out, out[label_col])):
        folds[val_idx] = fold
    if (folds < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("StratifiedKFold left rows unassigned")
    out["fold"] = folds
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_folds(
    assignment: Dict[str, int],
    path: str | os.PathLike[str],
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Write ``{"n_folds":…, "seed":…, "folds": {filename: fold}}`` as JSON."""
    payload = {
        "n_folds": int(n_folds),
        "seed": int(seed),
        "folds": {str(k): int(v) for k, v in assignment.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    LOG.info("Wrote fold assignment (%d files) -> %s", len(assignment), path)


def load_folds(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Read the JSON written by :func:`save_folds` (``{}`` when absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Could not read folds file %s (%s); regenerating.", path, exc)
        return {}
    if not isinstance(payload, dict) or "folds" not in payload:
        LOG.warning("Malformed folds file %s; regenerating.", path)
        return {}
    return payload


def load_or_create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    path: str | os.PathLike[str] = "data/folds.json",
    label_col: str = "emotion",
) -> pd.DataFrame:
    """Attach a ``fold`` column, reusing ``path`` when it matches the data.

    The cached assignment is reused only when *all* of the following hold:
    same ``n_folds``, same ``seed``, and exactly the same set of file names.
    Otherwise folds are regenerated (and the cache rewritten) with a log
    message explaining why.
    """
    if "filename" not in df.columns:
        raise ValueError("df must contain a 'filename' column")

    cached = load_folds(path)
    files = [str(f) for f in df["filename"].tolist()]
    reuse = False
    reason = "no cache"
    if cached:
        cached_map: Dict[str, int] = {str(k): int(v) for k, v in cached["folds"].items()}
        stale = set(cached_map) ^ set(files)
        if int(cached.get("n_folds", -1)) != int(n_folds):
            reason = f"n_folds changed ({cached.get('n_folds')} -> {n_folds})"
        elif int(cached.get("seed", -1)) != int(seed):
            reason = f"seed changed ({cached.get('seed')} -> {seed})"
        elif stale:
            reason = f"{len(stale)} file name(s) differ (e.g. {sorted(stale)[:3]})"
        else:
            reuse = True

    if reuse:
        LOG.info("Reusing fold assignment from %s", path)
        out = df.copy().reset_index(drop=True)
        out["fold"] = [int(cached["folds"][str(f)]) for f in out["filename"]]
        return out

    LOG.info("Creating new stratified %d-fold split (%s)", int(n_folds), reason)
    out = make_stratified_folds(df, n_folds=n_folds, seed=seed, label_col=label_col)
    try:
        save_folds(
            dict(zip(out["filename"].astype(str), out["fold"])),
            path,
            n_folds=n_folds,
            seed=seed,
        )
    except OSError as exc:
        LOG.warning("Could not persist folds to %s (%s); continuing in-memory.", path, exc)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fold_split(
    df: pd.DataFrame, fold: int, fold_col: str = "fold"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_part, val_part)`` for one fold (index reset)."""
    if fold_col not in df.columns:
        raise ValueError(f"df has no {fold_col!r} column; call load_or_create_folds first")
    train = df.loc[df[fold_col] != int(fold)].reset_index(drop=True)
    val = df.loc[df[fold_col] == int(fold)].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(f"fold {fold} produced an empty train/val split")
    return train, val


def fold_assignment_path(data_root: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> str:
    """Pick a writable location for the folds cache.

    ``data_root`` is read-only on Kaggle (``/kaggle/input/...``), so fall back
    to ``output_dir`` there.
    """
    data_root = Path(data_root)
    candidate = data_root / "folds.json"
    try:
        if data_root.exists() and os.access(data_root, os.W_OK):
            return str(candidate)
    except OSError:
        pass
    return str(Path(output_dir) / "folds.json")
