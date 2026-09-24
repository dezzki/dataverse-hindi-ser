"""CSV readers and audio loading utilities.

Audio contract used across the pipeline: **float32 mono** numpy array sampled
at ``cfg.sample_rate`` with a fixed length of ``cfg.sample_rate * cfg.max_seconds``
samples (zero-padded if shorter, centre/random-cropped if longer).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import EMOTIONS, EMOTION_TO_IDX, ExperimentConfig  # noqa: F401  (re-export)

LOG = logging.getLogger(__name__)

__all__ = [
    "EMOTIONS",
    "EMOTION_TO_IDX",
    "read_train_df",
    "read_test_df",
    "resolve_audio_path",
    "load_audio_raw",
    "fix_length",
    "load_audio",
    "random_crop",
]


# --------------------------------------------------------------------------- #
# CSV readers
# --------------------------------------------------------------------------- #
def _normalise_filename(series: pd.Series) -> pd.Series:
    """Strip directories/whitespace, keep just the audio file name."""
    return (
        series.astype(str)
        .str.strip()
        .map(lambda s: os.path.basename(s.replace("\\", "/")))
    )


def _id_column(df: pd.DataFrame, name: str = "Id") -> pd.Series:
    """Return an integer ``Id`` column when possible (ids are numeric here)."""
    if name not in df.columns:
        return pd.Series(np.arange(len(df)), name="Id")
    col = df[name]
    try:
        return col.astype("int64").rename("Id")
    except (TypeError, ValueError):
        return col.astype(str).rename("Id")


def _suggest_data_roots(base: str | os.PathLike[str], max_depth: int = 2) -> list:
    """Find likely ``data_root`` directories containing train.csv/test.csv.

    Only used to make "file not found" errors actionable (e.g. the zip was
    extracted into ``data/<competition>/``).
    """
    base = Path(base)
    if not base.exists():
        return []
    found: list = []
    try:
        candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
        for cand in candidates:
            if (cand / "train.csv").exists() or (cand / "test.csv").exists():
                found.append(str(cand))
            if max_depth > 1:
                for sub in sorted(p for p in cand.iterdir() if p.is_dir()):
                    if (sub / "train.csv").exists() or (sub / "test.csv").exists():
                        found.append(str(sub))
    except OSError:
        return found
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _not_found_message(kind: str, csv_path: Path, root: str | os.PathLike[str]) -> str:
    message = f"{kind} CSV not found: {csv_path}."
    suggestions = _suggest_data_roots(root)
    if suggestions:
        message += f" Found candidate data_root(s) instead: {suggestions}. "
    return message + " Set data_root correctly (local: 'data', Kaggle: '/kaggle/input/<competition>')."


def read_train_df(root: str | os.PathLike[str], cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    """Read ``<root>/train.csv`` -> DataFrame[Id, filename, emotion, path].

    Labels are lower-cased/stripped; rows whose label is not one of
    ``EMOTIONS`` are dropped with a loud warning (they cannot be trained on).
    """
    cfg = cfg or ExperimentConfig()
    csv_path = Path(root) / cfg.train_csv
    if not csv_path.exists():
        raise FileNotFoundError(_not_found_message("Training", csv_path, root))
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'filename' column, got {list(df.columns)}")
    if "emotion" not in df.columns:
        raise ValueError(f"{csv_path} must contain an 'emotion' column, got {list(df.columns)}")

    out = pd.DataFrame(
        {
            "Id": _id_column(df),
            "filename": _normalise_filename(df["filename"]),
            "emotion": df["emotion"].astype(str).str.strip().str.lower(),
        }
    )

    bad = ~out["emotion"].isin(EMOTIONS)
    if bad.any():
        offenders = sorted(out.loc[bad, "emotion"].unique().tolist())
        LOG.warning(
            "Dropping %d/%d train rows with labels outside %s (found: %s)",
            int(bad.sum()),
            len(out),
            list(EMOTIONS),
            offenders,
        )
        out = out.loc[~bad].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable training rows in {csv_path} (labels must be in {list(EMOTIONS)})")

    out["path"] = [resolve_audio_path(root, f, split="train", cfg=cfg) for f in out["filename"]]
    missing = out["path"].isna()
    if missing.any():
        LOG.warning(
            "%d train audio files referenced by %s were not found (first: %s)",
            int(missing.sum()),
            csv_path,
            out.loc[missing, "filename"].head(3).tolist(),
        )
    return out


def read_test_df(root: str | os.PathLike[str], cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    """Read ``<root>/test.csv`` -> DataFrame[Id, filename, path]."""
    cfg = cfg or ExperimentConfig()
    csv_path = Path(root) / cfg.test_csv
    if not csv_path.exists():
        raise FileNotFoundError(_not_found_message("Test", csv_path, root))
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'filename' column, got {list(df.columns)}")

    out = pd.DataFrame(
        {"Id": _id_column(df), "filename": _normalise_filename(df["filename"])}
    )
    out["path"] = [resolve_audio_path(root, f, split="test", cfg=cfg) for f in out["filename"]]
    missing = out["path"].isna()
    if missing.any():
        LOG.warning(
            "%d test audio files referenced by %s were not found (first: %s)",
            int(missing.sum()),
            csv_path,
            out.loc[missing, "filename"].head(3).tolist(),
        )
    return out


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def resolve_audio_path(
    root: str | os.PathLike[str],
    filename: str,
    split: str = "train",
    cfg: Optional[ExperimentConfig] = None,
) -> Optional[str]:
    """Locate an audio file under ``root`` for the given split.

    Tries (in order): the split sub-directory, the root itself, and any
    ``audio/`` sibling — because Kaggle archives differ in layout.
    Returns ``None`` when nothing is found (callers decide how loud to be).
    """
    cfg = cfg or ExperimentConfig()
    root = Path(root)
    name = os.path.basename(str(filename).replace("\\", "/"))
    subdir = cfg.train_audio_dir if split == "train" else cfg.test_audio_dir
    candidates = [
        root / subdir / name,
        root / str(filename),
        root / name,
        root / "audio" / name,
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio_raw(
    path: str | os.PathLike[str],
    sr: int = 16_000,
    trim: bool = False,
    top_db: float = 40.0,
    min_seconds: float = 0.15,
) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sr`` Hz (no fixed length).

    ``trim=True`` removes leading/trailing near-silence with a conservative
    ``librosa.effects.trim(top_db=...)`` threshold; a short guard prevents
    trimming a whole utterance down to nothing.
    """
    import librosa  # local import: keeps module import light

    y, _ = librosa.load(str(path), sr=int(sr), mono=True, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Decoded empty audio: {path}")
    if trim:
        try:
            trimmed, _ = librosa.effects.trim(y, top_db=float(top_db))
            if trimmed.size >= int(min_seconds * sr):
                y = trimmed
            else:
                LOG.debug("Skipping silence trim for %s (would leave %d samples)", path, trimmed.size)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("Silence trim failed for %s: %s", path, exc)
    return np.ascontiguousarray(y, dtype=np.float32)


def fix_length(y: np.ndarray, n_samples: int, mode: str = "center", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pad (zeros) or crop ``y`` to exactly ``n_samples`` samples.

    ``mode``: ``center`` | ``random`` | ``start`` (crop anchor when too long).
    """
    n = int(n_samples)
    y = np.asarray(y, dtype=np.float32)
    if y.size == n:
        return np.ascontiguousarray(y)
    if y.size > n:
        if mode == "start":
            start = 0
        elif mode == "random":
            start = int(rng.integers(0, y.size - n + 1)) if rng is not None else (y.size - n) // 2
        else:  # center
            start = (y.size - n) // 2
        y = y[start : start + n]
    else:
        pad_left = (n - y.size) // 2 if mode != "start" else 0
        y = np.pad(y, (pad_left, n - y.size - pad_left), mode="constant")
    return np.ascontiguousarray(y, dtype=np.float32)


def random_crop(y: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop / zero pad to ``n_samples`` (never raises on short input)."""
    return fix_length(y, n_samples, mode="random", rng=rng)


def load_audio(
    path: str | os.PathLike[str],
    sr: int = 16_000,
    max_secs: float = 6.0,
    trim: bool = False,
    top_db: float = 40.0,
    mode: str = "center",
    rng: Optional[np.random.Generator] = None,
    allow_missing: bool = False,
) -> np.ndarray:
    """Load + resample + (optionally) trim + pad/truncate to ``sr * max_secs``.

    Returns a ``float32`` array of shape ``(sr * max_secs,)``.
    """
    n_samples = int(round(sr * max_secs))
    if path is None or (isinstance(path, (str, os.PathLike)) and not os.path.exists(str(path))):
        if not allow_missing:
            raise FileNotFoundError(f"Audio file not found: {path}")
        LOG.warning("Returning silence for missing audio: %s", path)
        return np.zeros(n_samples, dtype=np.float32)
    y = load_audio_raw(path, sr=sr, trim=trim, top_db=top_db)
    return fix_length(y, n_samples, mode=mode, rng=rng)


def load_audio_for(
    row_or_path: Any,
    cfg: ExperimentConfig,
    mode: str = "center",
    rng: Optional[np.random.Generator] = None,
    trim: Optional[bool] = None,
    max_secs: Optional[float] = None,
) -> np.ndarray:
    """Convenience wrapper: load audio for a DataFrame row or a path."""
    path = row_or_path.get("path") if hasattr(row_or_path, "get") else row_or_path
    return load_audio(
        path,
        sr=cfg.sample_rate,
        max_secs=cfg.max_seconds if max_secs is None else max_secs,
        trim=cfg.trim_silence if trim is None else trim,
        top_db=cfg.trim_top_db,
        mode=mode,
        rng=rng,
    )


def split_summary(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> str:
    """Human-readable class/split census (used in logs and reports)."""
    counts: Dict[str, int] = train_df["emotion"].value_counts().to_dict()
    lines = [
        f"train rows: {len(train_df)}",
        "class counts: " + ", ".join(f"{e}={counts.get(e, 0)}" for e in EMOTIONS),
    ]
    if test_df is not None:
        lines.append(f"test rows: {len(test_df)}")
    return " | ".join(lines)
