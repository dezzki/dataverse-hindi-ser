"""Classical feature extraction: eGeMAPS (openSMILE) and MFCC statistics.

Both extractors return a ``DataFrame`` with a leading ``filename`` column and
numeric feature columns, so they can be merged on ``filename``:

    merged = pd.merge(extract_opensmile(df, root, cfg), extract_mfcc_stats(df, root, cfg), on="filename")

``opensmile`` is imported lazily and a missing install raises a clear error with
the pip command; importing this module never requires it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_audio_raw, resolve_audio_path

LOG = logging.getLogger(__name__)

__all__ = ["extract_opensmile", "extract_mfcc_stats", "build_classic_features"]

_EGEMAPS_CANDIDATES = ("eGeMAPSv02", "eGeMAPSv01b", "ComParE_2016")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _row_paths(df: pd.DataFrame, root: str, split: str, cfg: ExperimentConfig) -> List[Optional[str]]:
    if "path" in df.columns:
        return [str(p) if isinstance(p, str) and p else None for p in df["path"].tolist()]
    return [resolve_audio_path(root, f, split=split, cfg=cfg) for f in df["filename"].astype(str)]


def _cache_file(out_dir: str, name: str, df: pd.DataFrame, params: Dict[str, Any], enabled: bool) -> Optional[str]:
    if not enabled:
        return None
    payload = json.dumps(
        {"name": name, "params": params, "files": [str(f) for f in df["filename"].tolist()]},
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return os.path.join(out_dir, "features", f"{name}_{digest}.pkl")


def _progress(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, disable=None)
    except Exception:  # pragma: no cover - tqdm absent
        return iterable


# --------------------------------------------------------------------------- #
# openSMILE
# --------------------------------------------------------------------------- #
def _make_smile(feature_set: str, sample_rate: int):
    import opensmile

    try:
        fs = getattr(opensmile.FeatureSet, feature_set)
    except AttributeError as exc:  # pragma: no cover - version drift
        raise RuntimeError(
            f"opensmile.FeatureSet.{feature_set} not available in this opensmile build"
        ) from exc
    return opensmile.Smile(
        feature_set=fs,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=int(sample_rate),
        resample=True,
    )


def extract_opensmile(
    df: pd.DataFrame,
    root: str = "data",
    cfg: Optional[ExperimentConfig] = None,
    split: str = "train",
    use_cache: Optional[bool] = None,
    feature_set: Optional[str] = None,
) -> pd.DataFrame:
    """Extract functionals (eGeMAPS by default, ComParE as fallback) per file.

    Raises
    ------
    RuntimeError
        If the ``opensmile`` package is not installed (message says how to fix).
    """
    cfg = cfg or ExperimentConfig()
    features_cfg = cfg.features if isinstance(cfg.features, dict) else {}
    use_cache = bool(features_cfg.get("cache", True)) if use_cache is None else bool(use_cache)
    candidates = [feature_set] if feature_set else list(_EGEMAPS_CANDIDATES)
    params = {"sample_rate": cfg.sample_rate, "max_seconds": cfg.max_seconds, "candidates": candidates}

    cache_path = _cache_file(cfg.output_dir, "opensmile", df, params, use_cache)
    if cache_path and os.path.exists(cache_path):
        LOG.info("Loading cached openSMILE features: %s", cache_path)
        return pd.read_pickle(cache_path)

    try:
        import opensmile  # noqa: F401  (probe only; real import inside _make_smile)
    except ImportError as exc:
        raise RuntimeError(
            "The 'opensmile' package is required for eGeMAPS features. "
            "Install it with:  pip install opensmile   (or: uv pip install opensmile)"
        ) from exc
    del opensmile

    smile = None
    used_set = None
    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            smile = _make_smile(name, cfg.sample_rate)
            used_set = name
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            LOG.warning("openSMILE feature set %s unavailable: %s", name, exc)
    if smile is None:
        raise RuntimeError(f"No usable openSMILE feature set among {candidates}: {last_error}")

    n_samples = int(cfg.sample_rate * cfg.max_seconds)
    paths = _row_paths(df, root, split, cfg)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for filename, path in _progress(list(zip(df["filename"].astype(str), paths)), len(df), f"opensmile:{used_set}"):
        if path is None:
            skipped += 1
            rows.append({"filename": filename})
            continue
        try:
            feats = smile.process_file(str(path))
            series = feats.iloc[0] if isinstance(feats, pd.DataFrame) else feats
            record = {"filename": filename}
            for key, value in series.items():
                col = key[0] if isinstance(key, tuple) else str(key)
                record[f"osm_{col}"] = float(value)
            rows.append(record)
        except Exception as exc:  # pragma: no cover - per-file decode issues
            skipped += 1
            LOG.warning("openSMILE failed on %s: %s", filename, exc)
            rows.append({"filename": filename})

    out = pd.DataFrame(rows).fillna(0.0)
    if skipped:
        LOG.warning("openSMILE: %d/%d files could not be processed (features zero-filled)", skipped, len(df))
    LOG.info("openSMILE %s -> %s", used_set, tuple(out.shape))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_pickle(cache_path)
        LOG.info("Cached openSMILE features -> %s", cache_path)
    del n_samples  # only kept for symmetry/documentation with the MFCC path
    return out


# --------------------------------------------------------------------------- #
# MFCC statistics
# --------------------------------------------------------------------------- #
def extract_mfcc_stats(
    df: pd.DataFrame,
    root: str = "data",
    cfg: Optional[ExperimentConfig] = None,
    split: str = "train",
    use_cache: Optional[bool] = None,
) -> pd.DataFrame:
    """MFCC ``[mean, std, min, max]`` over time -> ``4 * n_mfcc`` features."""
    import librosa

    cfg = cfg or ExperimentConfig()
    features_cfg = cfg.features if isinstance(cfg.features, dict) else {}
    use_cache = bool(features_cfg.get("cache", True)) if use_cache is None else bool(use_cache)
    n_mfcc = int(features_cfg.get("mfcc_n", 40))
    n_fft = int(features_cfg.get("n_fft", 1024))
    hop = int(features_cfg.get("hop_length", 256))
    params = {
        "sample_rate": cfg.sample_rate,
        "max_seconds": cfg.max_seconds,
        "n_mfcc": n_mfcc,
        "n_fft": n_fft,
        "hop_length": hop,
        "trim": cfg.trim_silence,
    }

    cache_path = _cache_file(cfg.output_dir, "mfcc", df, params, use_cache)
    if cache_path and os.path.exists(cache_path):
        LOG.info("Loading cached MFCC features: %s", cache_path)
        return pd.read_pickle(cache_path)

    paths = _row_paths(df, root, split, cfg)
    rows: List[Dict[str, Any]] = []
    stats = ("mean", "std", "min", "max")
    for filename, path in _progress(list(zip(df["filename"].astype(str), paths)), len(df), "mfcc-stats"):
        record: Dict[str, Any] = {"filename": filename}
        try:
            y = load_audio_raw(path, sr=cfg.sample_rate, trim=cfg.trim_silence, top_db=cfg.trim_top_db) if path else np.zeros(
                int(cfg.sample_rate * cfg.max_seconds), dtype=np.float32
            )
            mfcc = librosa.feature.mfcc(y=y, sr=int(cfg.sample_rate), n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
            agg = {
                "mean": mfcc.mean(axis=1),
                "std": mfcc.std(axis=1),
                "min": mfcc.min(axis=1),
                "max": mfcc.max(axis=1),
            }
            for stat in stats:
                for i, value in enumerate(np.asarray(agg[stat]).ravel()):
                    record[f"mfcc_{stat}_{i:02d}"] = float(value)
        except Exception as exc:  # pragma: no cover - per-file decode issues
            LOG.warning("MFCC failed on %s: %s", filename, exc)
        rows.append(record)

    out = pd.DataFrame(rows).fillna(0.0)
    LOG.info("MFCC stats (n_mfcc=%d) -> %s", n_mfcc, tuple(out.shape))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_pickle(cache_path)
        LOG.info("Cached MFCC features -> %s", cache_path)
    return out


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #
def build_classic_features(
    df: pd.DataFrame,
    root: str = "data",
    cfg: Optional[ExperimentConfig] = None,
    split: str = "train",
) -> pd.DataFrame:
    """Concatenate the enabled feature blocks into one numeric matrix.

    Returns a DataFrame with a ``filename`` column plus all feature columns.
    """
    cfg = cfg or ExperimentConfig()
    features_cfg = cfg.features if isinstance(cfg.features, dict) else {}
    blocks: List[pd.DataFrame] = []
    if features_cfg.get("opensmile", True):
        blocks.append(extract_opensmile(df, root, cfg, split=split))
    if features_cfg.get("mfcc", True):
        blocks.append(extract_mfcc_stats(df, root, cfg, split=split))
    if not blocks:
        raise ValueError("cfg.features enables neither 'opensmile' nor 'mfcc'")

    merged = blocks[0]
    for block in blocks[1:]:
        merged = pd.merge(merged, block, on="filename", how="outer")
    feature_cols = [c for c in merged.columns if c != "filename"]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    LOG.info("Classic feature matrix: %d rows x %d features", len(merged), len(feature_cols))
    return merged
