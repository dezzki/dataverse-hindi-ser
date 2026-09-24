"""Waveform / spectrogram augmentation, all config-driven and seeded.

Design notes
------------
* Every op is a no-op when disabled (``enabled: false``) or when the config
  key is a plain ``False``.
* Randomness comes from a per-instance ``numpy.random.Generator``; call
  :meth:`WaveAug.reseed` before each item (the dataset does this) so results
  are reproducible regardless of DataLoader worker count.
* Spectrogram SpecAugment is applied on ``(..., n_mels, time)`` torch tensors
  and has its own ``torch.Generator``.
* :func:`mixup` returns **soft one-hot targets** ready for a soft-target CE.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .config import DEFAULT_AUG

LOG = logging.getLogger(__name__)

__all__ = ["WaveAug", "mixup", "normalise_aug_config"]


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def normalise_aug_config(aug_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalise the ``aug`` block into ``{op: {enabled: bool, ...params}}``.

    Accepts ``{"speed": False}``, ``{"speed": True}`` or full dicts, and fills
    missing parameters from :data:`src.config.DEFAULT_AUG`. A top-level
    ``enabled: false`` (or passing a bool / ``None`` instead of a mapping)
    disables every op.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(aug_cfg, bool):
        aug_cfg = {"enabled": aug_cfg}
    aug_cfg = dict(aug_cfg or {})
    global_enabled = bool(aug_cfg.pop("enabled", True))
    for op, defaults in DEFAULT_AUG.items():
        value = aug_cfg.get(op, defaults)
        if isinstance(value, bool):
            merged = dict(defaults)
            merged["enabled"] = bool(value)
        elif isinstance(value, Mapping):
            merged = {**defaults, **dict(value)}
            merged["enabled"] = bool(merged.get("enabled", True))
        elif value is None:
            merged = {**defaults, "enabled": False}
        else:
            raise TypeError(f"aug[{op!r}] must be a bool or mapping, got {type(value).__name__}")
        if not global_enabled:
            merged["enabled"] = False
        out[op] = merged
    return out


def _cfg_attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from an object attribute or a mapping."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --------------------------------------------------------------------------- #
# Waveform augmentation
# --------------------------------------------------------------------------- #
class WaveAug:
    """Config-driven waveform augmenter.

    Parameters
    ----------
    aug_cfg or cfg:
        Either an ``ExperimentConfig`` (``aug`` block + ``sample_rate`` +
        ``seed`` are read from it) or the raw ``aug`` mapping.
    sample_rate, seed, enabled, max_seconds:
        Explicit overrides when a raw mapping is passed.
    """

    def __init__(
        self,
        aug_cfg: Optional[Mapping[str, Any]] = None,
        sample_rate: int = 16_000,
        seed: int = 42,
        enabled: bool = True,
        max_seconds: float = 6.0,
    ) -> None:
        raw = _cfg_attr(aug_cfg, "aug", aug_cfg)
        if isinstance(aug_cfg, Mapping) and "aug" in aug_cfg and isinstance(aug_cfg["aug"], Mapping):
            raw = aug_cfg["aug"]
        self.sample_rate = int(_cfg_attr(aug_cfg, "sample_rate", sample_rate))
        self.seed = int(_cfg_attr(aug_cfg, "seed", seed))
        self.max_seconds = float(_cfg_attr(aug_cfg, "max_seconds", max_seconds))
        self.ops = normalise_aug_config(raw)
        if not enabled:
            for op in self.ops.values():
                op["enabled"] = False
        self.rng = np.random.default_rng(self.seed)
        self.torch_gen = torch.Generator().manual_seed(self.seed)

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return any(op.get("enabled", False) for op in self.ops.values())

    def reseed(self, seed: int) -> None:
        """Deterministically re-seed both RNGs (used per dataset item/epoch)."""
        seed = int(seed) % (2**32 - 1)
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator().manual_seed(seed)

    def describe(self) -> str:
        on = [name for name, op in self.ops.items() if op.get("enabled")]
        return "WaveAug(" + (", ".join(on) if on else "disabled") + ")"

    # ------------------------------------------------------------------ #
    # Individual ops
    # ------------------------------------------------------------------ #
    def time_stretch(self, y: np.ndarray) -> np.ndarray:
        import librosa

        op = self.ops["speed"]
        rate = float(self.rng.uniform(op["min_rate"], op["max_rate"]))
        try:
            y2 = librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("time_stretch failed (rate=%.3f): %s", rate, exc)
            return y
        return np.ascontiguousarray(y2, dtype=np.float32)

    def pitch_shift(self, y: np.ndarray) -> np.ndarray:
        import librosa

        op = self.ops["pitch"]
        steps = float(self.rng.uniform(op["min_steps"], op["max_steps"]))
        try:
            y2 = librosa.effects.pitch_shift(
                np.asarray(y, dtype=np.float32), sr=self.sample_rate, n_steps=steps
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("pitch_shift failed (steps=%.3f): %s", steps, exc)
            return y
        return np.ascontiguousarray(y2, dtype=np.float32)

    def add_noise(self, y: np.ndarray) -> np.ndarray:
        op = self.ops["noise"]
        snr_db = float(self.rng.uniform(op["min_snr_db"], op["max_snr_db"]))
        y = np.asarray(y, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
        if rms <= 0:
            return y
        noise = self.rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
        target_noise_rms = rms / (10.0 ** (snr_db / 20.0))
        return np.ascontiguousarray(y + noise * (target_noise_rms / noise_rms), dtype=np.float32)

    def spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """Frequency + time masking on a ``(..., n_mels, time)`` tensor."""
        op = self.ops["specaug"]
        mel = mel.clone()
        n_mels, n_frames = int(mel.shape[-2]), int(mel.shape[-1])
        for _ in range(int(op["num_freq_masks"])):
            width = int(self._randint(1, int(op["freq_mask_width"]) + 1))
            if width <= 0 or n_mels <= 1:
                continue
            width = min(width, n_mels)
            start = int(self._randint(0, n_mels - width + 1))
            mel[..., start : start + width, :] = 0.0
        for _ in range(int(op["num_time_masks"])):
            width = int(self._randint(1, int(op["time_mask_width"]) + 1))
            if width <= 0 or n_frames <= 1:
                continue
            width = min(width, n_frames)
            start = int(self._randint(0, n_frames - width + 1))
            mel[..., start : start + width] = 0.0
        return mel

    #: alias kept for readability at call sites
    spec_augment_logmel = spec_augment

    def _randint(self, low: int, high: int) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high, (1,), generator=self.torch_gen).item())

    def _maybe(self, op_name: str) -> bool:
        op = self.ops[op_name]
        if not op.get("enabled", False):
            return False
        return bool(self.rng.random() < float(op.get("prob", 1.0)))

    # ------------------------------------------------------------------ #
    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Apply the enabled waveform ops in a fixed order."""
        if not self.enabled:
            return np.asarray(y, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if self._maybe("speed"):
            y = self.time_stretch(y)
        if self._maybe("pitch"):
            y = self.pitch_shift(y)
        if self._maybe("noise"):
            y = self.add_noise(y)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Mixup
# --------------------------------------------------------------------------- #
def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    num_classes: int = 4,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup for waveform batches.

    Parameters
    ----------
    x: ``(B, C, T)`` float tensor.
    y: ``(B,)`` int64 labels.
    alpha: Beta parameter; ``<= 0`` disables mixing.
    num_classes: size of the soft target vectors.

    Returns
    -------
    ``(x_mixed, soft_targets)`` with ``soft_targets`` of shape ``(B, num_classes)``.
    """
    onehot = torch.nn.functional.one_hot(y.long(), num_classes=num_classes).float()
    if alpha is None or alpha <= 0 or x.size(0) < 2:
        return x, onehot

    if generator is None:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(x.size(0))
    else:
        lam = float(
            torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
            .sample()
            .item()
        )
        perm = torch.randperm(x.size(0), generator=generator)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    targets = lam * onehot + (1.0 - lam) * onehot[perm]
    return x_mixed, targets
