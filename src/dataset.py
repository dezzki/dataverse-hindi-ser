"""Torch ``Dataset`` for SER: yields ``(waveform[1, T], label)``.

Loaded waveforms are cached in a module-level dict (LRU-ish: cleared when it
grows too large) so that repeated epochs / multi-crop inference do not re-decode
the same file. Each DataLoader worker gets its own copy of the cache.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import WaveAug
from .config import EMOTION_TO_IDX, ExperimentConfig
from .data import fix_length

LOG = logging.getLogger(__name__)

__all__ = ["SERDataset", "collate_waveforms", "get_cache", "clear_cache"]

#: path -> float32 waveform (full length, before crop/pad)
_WAVE_CACHE: Dict[str, np.ndarray] = {}
_MAX_CACHE_ITEMS = 4096


def get_cache() -> Dict[str, np.ndarray]:
    return _WAVE_CACHE


def clear_cache() -> None:
    _WAVE_CACHE.clear()


def _cache_put(key: str, value: np.ndarray) -> None:
    if len(_WAVE_CACHE) >= _MAX_CACHE_ITEMS:
        LOG.debug("Audio cache full (%d items) - clearing", len(_WAVE_CACHE))
        _WAVE_CACHE.clear()
    _WAVE_CACHE[key] = value


class SERDataset(Dataset):
    """Dataset over a DataFrame produced by :mod:`src.data` / :mod:`src.splits`.

    Parameters
    ----------
    df:
        Rows with at least ``filename``; ``path`` is used when present, and
        ``emotion`` for labels (test rows may omit it, label becomes ``-1``).
    root:
        ``data_root``; used only when the frame has no ``path`` column.
    cfg:
        :class:`~src.config.ExperimentConfig`.
    fold:
        When given together with a ``fold`` column: ``split='train'`` keeps the
        other folds, ``split='val'`` keeps only this fold. ``split='test'``
        ignores folds entirely.
    split:
        ``train`` | ``val`` | ``test`` (also selects the audio sub-directory).
    augment:
        Apply :class:`~src.augment.WaveAug` (and random cropping).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | os.PathLike[str],
        cfg: ExperimentConfig,
        fold: Optional[int] = None,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.cfg = cfg
        self.split = split
        self.fold = None if fold is None else int(fold)
        self.root = str(root)
        self.augment = bool(augment) and split == "train"
        self.n_samples = int(cfg.max_samples)

        frame = df
        if split in ("train", "val") and self.fold is not None and "fold" in frame.columns:
            if split == "train":
                frame = frame.loc[frame["fold"] != self.fold]
            else:
                frame = frame.loc[frame["fold"] == self.fold]
        self.df = frame.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Empty {split} split for fold {fold} (len(df)=0)")

        self.filenames: List[str] = [str(f) for f in self.df["filename"].tolist()]
        self.folds: List[int] = (
            [int(f) for f in self.df["fold"].tolist()] if "fold" in self.df.columns else [-1] * len(self.df)
        )
        self.labels: List[int] = (
            [int(EMOTION_TO_IDX[e]) for e in self.df["emotion"].astype(str).str.lower()]
            if "emotion" in self.df.columns
            else [-1] * len(self.df)
        )
        self.paths: List[Optional[str]] = (
            [str(p) if isinstance(p, str) else None for p in self.df["path"].tolist()]
            if "path" in self.df.columns
            else [None] * len(self.df)
        )
        self.aug = WaveAug(cfg, enabled=self.augment) if self.augment else None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.df)

    def _load(self, idx: int) -> np.ndarray:
        """Decode (or fetch from cache) the waveform of item ``idx``."""
        cfg = self.cfg
        path = self.paths[idx]
        if path is None:
            from .data import resolve_audio_path

            path = resolve_audio_path(self.root, self.filenames[idx], split=self.split, cfg=cfg)
            self.paths[idx] = path
        key = str(path)
        cached = _WAVE_CACHE.get(key)
        if cached is not None:
            return cached
        from .data import load_audio_raw

        if path is None:
            LOG.warning("Missing audio for %s - using silence", self.filenames[idx])
            wave = np.zeros(int(cfg.sample_rate * max(cfg.max_seconds, 0.1)), dtype=np.float32)
        else:
            wave = load_audio_raw(path, sr=cfg.sample_rate, trim=cfg.trim_silence, top_db=cfg.trim_top_db)
        _cache_put(key, wave)
        return wave

    def _item_seed(self, idx: int, epoch: int = 0) -> int:
        """Deterministic per-item seed (worker-count independent)."""
        base = (int(self.cfg.seed) * 1_000_003 + int(epoch) * 7_919 + int(idx)) % (2**31 - 1)
        return int(base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cfg = self.cfg
        wave = self._load(int(idx))
        if self.augment and self.aug is not None:
            # Seed per item so workers don't duplicate the same random stream.
            self.aug.reseed(self._item_seed(idx, getattr(self, "_epoch", 0)))
            wave = self.aug(wave)
            wave = fix_length(wave, self.n_samples, mode="random", rng=self.aug.rng)
        else:
            wave = fix_length(wave, self.n_samples, mode="center")
        tensor = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).unsqueeze(0)
        return tensor, int(self.labels[idx])

    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int) -> None:
        """Bump the epoch so augmentation randomness changes between epochs."""
        self._epoch = int(epoch)

    def label_array(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int64)

    def class_counts(self) -> Dict[str, int]:
        from .config import IDX_TO_EMOTION

        counts: Dict[str, int] = {}
        for y in self.labels:
            key = IDX_TO_EMOTION.get(int(y), "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def collate_waveforms(batch: List[Tuple[torch.Tensor, int]]):
    """Pad waveforms within a batch to the longest one (zero padding).

    Returns ``(waves[B, 1, T_max], labels[B], lengths[B])``. Intended to be used
    as ``DataLoader(collate_fn=...)``.
    """
    waves, labels = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
    max_len = int(lengths.max().item())
    channels = waves[0].shape[0]
    out = torch.zeros(len(waves), channels, max_len, dtype=torch.float32)
    for i, w in enumerate(waves):
        out[i, :, : w.shape[-1]] = w
    return out, torch.tensor(labels, dtype=torch.long), lengths


def make_dataloader(
    dataset: SERDataset,
    cfg: ExperimentConfig,
    shuffle: bool = False,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    drop_last: bool = False,
) -> Any:
    """Build a ``DataLoader`` with the pipeline's collate function."""
    from torch.utils.data import DataLoader

    nw = cfg.num_workers if num_workers is None else int(num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg.batch_size),
        shuffle=bool(shuffle),
        num_workers=nw,
        pin_memory=True,
        drop_last=bool(drop_last),
        collate_fn=collate_waveforms,
        persistent_workers=bool(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
