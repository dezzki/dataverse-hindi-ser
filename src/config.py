"""Experiment configuration: dataclass + YAML loading.

This module intentionally depends only on the standard library + PyYAML so it
can be imported in lightweight contexts (notebooks, feature scripts) without
torch/librosa installed.

Usage
-----
>>> from src.config import load_config
>>> cfg = load_config("configs/xlsr.yaml")
>>> cfg.model_name
'facebook/wav2vec2-large-xlsr-53'

CLI (debug)::

    python -m src.config --config configs/xlsr.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Label space (competition-defined, lowercase, this exact order is used
# everywhere probabilities are stored: angry, happy, neutral, sad).
# --------------------------------------------------------------------------- #
EMOTIONS: tuple = ("angry", "happy", "neutral", "sad")
EMOTION_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}
NUM_CLASSES: int = len(EMOTIONS)

#: Default augmentation block (fully config-driven; any value may be a bool
#: toggle, or a dict of params, or omitted).
DEFAULT_AUG: Dict[str, Any] = {
    "speed": {
        "enabled": True,
        "prob": 0.5,
        "min_rate": 0.9,
        "max_rate": 1.1,
    },
    "pitch": {
        "enabled": True,
        "prob": 0.5,
        "min_steps": -2.0,
        "max_steps": 2.0,
    },
    "noise": {
        "enabled": True,
        "prob": 0.5,
        "min_snr_db": 10.0,
        "max_snr_db": 30.0,
    },
    "specaug": {
        "enabled": True,
        "prob": 0.5,
        "num_freq_masks": 2,
        "freq_mask_width": 27,
        "num_time_masks": 2,
        "time_mask_width": 100,
    },
}

_VALID_POOLING = ("meanstd", "attention", "mean", "meanstd_attn")
_VALID_PRECISION = ("fp16", "bf16", "none", "no")
_VALID_SCHEDULERS = ("cosine", "linear", "constant", "onecycle", "none")


def _default_aug() -> Dict[str, Any]:
    """Deep-ish copy of the default augmentation block."""
    return {k: dict(v) for k, v in DEFAULT_AUG.items()}


@dataclass
class ExperimentConfig:
    """Everything the pipeline needs to run one experiment.

    ``data_root`` / ``output_dir`` are the only location anchors; every path
    elsewhere in the codebase is derived from them (or from CLI args), so the
    same configs work locally (``data/``) and on Kaggle
    (``/kaggle/input/<competition>/``).
    """

    # ----- model / architecture ------------------------------------------- #
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    #: pooling over the SSL time axis: ``meanstd`` | ``attention``
    pooling: str = "meanstd"
    #: freeze the first N transformer encoder layers (0 = train everything)
    freeze_encoder_layers: int = 0
    #: enable HF gradient checkpointing (memory <-> compute trade-off)
    grad_ckpt: bool = False
    #: optional explicit family override: ssl | whisper | auto
    model_family: str = "auto"
    #: hidden size of the 2-layer classification MLP
    head_hidden: int = 256
    dropout: float = 0.3

    # ----- audio ----------------------------------------------------------- #
    sample_rate: int = 16_000
    max_seconds: float = 6.0
    #: trim leading/trailing near-silence before padding/truncation
    trim_silence: bool = True
    #: librosa.effects.trim threshold in dB (conservative: only near-silence)
    trim_top_db: float = 40.0

    # ----- optimisation ---------------------------------------------------- #
    batch_size: int = 8
    grad_accum: int = 4
    #: base LR (used for the encoder, scaled by ``encoder_lr_scale``)
    lr: float = 1e-4
    #: LR for the classification head (falls back to ``lr`` when <= 0)
    head_lr: float = 1e-4
    #: encoder LR multiplier applied on top of ``lr``
    encoder_lr_scale: float = 0.1
    #: multiplicative decay per encoder layer going down (1.0 = uniform)
    layer_lr_decay: float = 0.95
    warmup_frac: float = 0.1
    epochs: int = 10
    #: 0 disables early stopping (train the full schedule)
    early_stop_patience: int = 3
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    mixed_precision: str = "fp16"  # fp16 | bf16 | none
    scheduler: str = "cosine"  # cosine | linear | constant | onecycle
    label_smoothing: float = 0.05
    #: BCE-style "turn CE into soft targets" mixup strength (0 disables)
    mixup_alpha: float = 0.2

    # ----- data / folds ---------------------------------------------------- #
    num_folds: int = 5
    folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    seed: int = 42
    num_workers: int = 2
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    train_audio_dir: str = "train"
    test_audio_dir: str = "test"
    data_root: str = "data"
    output_dir: str = "runs"
    #: persist OOF/test probabilities here (predict.py default)
    oof_dir: str = "oof"

    # ----- auxiliary (external) data ------------------------------------- #
    #: optional external CSV (same Id,filename,emotion format) added to the
    #: TRAIN split only (validation/OOF stays on the competition split).
    aux_csv: str = ""
    #: directory (or Kaggle input root) containing ``aux_csv`` + audio.
    aux_root: str = ""
    #: optional label remap {external_label: one_of(angry,happy,neutral,sad)}.
    aux_label_map: Dict[str, str] = field(default_factory=dict)

    # ----- augmentation / inference --------------------------------------- #
    aug: Dict[str, Any] = field(default_factory=_default_aug)
    #: number of random crops averaged at inference (used when ``tta`` is on)
    tta: bool = False
    tta_crops: int = 3

    # ----- classical baseline --------------------------------------------- #
    #: which classical models to fit (used by ``src.classic``)
    classic_models: List[str] = field(default_factory=lambda: ["sklearn_svm", "lightgbm"])
    #: feature toggles for ``src.classic``
    features: Dict[str, Any] = field(
        default_factory=lambda: {"opensmile": True, "mfcc": True, "cache": True}
    )
    #: free-form hyper-parameters for the classical models
    classic: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (JSON/YAML serialisable)."""
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def apply_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "ExperimentConfig":
        """Return a copy with ``overrides`` applied (unknown keys raise)."""
        if not overrides:
            return self
        known = {f.name for f in dataclasses.fields(self)}
        clean = {k: v for k, v in overrides.items() if k in known}
        unknown = set(overrides) - known
        if unknown:
            LOG.warning("Ignoring unknown config overrides: %s", sorted(unknown))
        return dataclasses.replace(self, **clean)  # type: ignore[arg-type]

    # ----- derived helpers ------------------------------------------------ #
    @property
    def sample_rate_f(self) -> float:
        return float(self.sample_rate)

    @property
    def max_samples(self) -> int:
        """Fixed waveform length (in samples) used across the pipeline."""
        return int(round(self.sample_rate * self.max_seconds))

    @property
    def precision(self) -> str:
        """Normalised mixed-precision string (``fp16``/``bf16``/``none``)."""
        p = str(self.mixed_precision).lower()
        return "none" if p == "no" else p

    def run_dir(self, tag: Optional[str] = None) -> str:
        """``<output_dir>/<tag>`` (tag defaults to the model basename)."""
        tag = tag or self.model_tag()
        return os.path.join(self.output_dir, tag)

    def model_tag(self) -> str:
        """Filesystem-safe short tag derived from ``model_name``."""
        base = self.model_name.rstrip("/").split("/")[-1]
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in base)

    def fold_dir(self, fold: int, tag: Optional[str] = None) -> str:
        return os.path.join(self.run_dir(tag), f"fold{int(fold)}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce_scalars(data: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-effort type coercion of raw YAML values to the dataclass types."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in spec:
            out[key] = value
            continue
        target = type(spec[key])
        try:
            if target is bool and not isinstance(value, bool):
                out[key] = str(value).strip().lower() in {"1", "true", "yes", "on"}
            elif target is int and not isinstance(value, bool):
                out[key] = int(value)
            elif target is float:
                out[key] = float(value)
            elif target is str:
                out[key] = str(value)
            elif target is list and isinstance(value, (list, tuple)):
                if spec[key] and isinstance(spec[key][0], int):
                    out[key] = [int(v) for v in value]
                else:
                    out[key] = list(value)
            else:
                out[key] = value
        except (TypeError, ValueError):  # keep the raw value; validation warns
            LOG.warning("Could not coerce config key %r=%r to %s", key, value, target)
            out[key] = value
    return out


def _validate(cfg: ExperimentConfig) -> ExperimentConfig:
    """Loud but non-fatal validation of the enum-ish fields."""
    if cfg.pooling not in _VALID_POOLING:
        raise ValueError(f"pooling must be one of {_VALID_POOLING}, got {cfg.pooling!r}")
    if str(cfg.mixed_precision).lower() not in _VALID_PRECISION:
        raise ValueError(
            f"mixed_precision must be one of {_VALID_PRECISION}, got {cfg.mixed_precision!r}"
        )
    if str(cfg.scheduler).lower() not in _VALID_SCHEDULERS:
        raise ValueError(
            f"scheduler must be one of {_VALID_SCHEDULERS}, got {cfg.scheduler!r}"
        )
    if cfg.num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {cfg.num_folds}")
    if cfg.max_seconds <= 0 or cfg.sample_rate <= 0:
        raise ValueError("sample_rate and max_seconds must be positive")
    if cfg.batch_size < 1 or cfg.grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")
    if not cfg.folds:
        cfg.folds = list(range(cfg.num_folds))
    if any(f < 0 or f >= cfg.num_folds for f in cfg.folds):
        LOG.warning(
            "cfg.folds=%s contains folds outside [0, %d); they will be ignored.",
            cfg.folds,
            cfg.num_folds,
        )
        cfg.folds = [f for f in cfg.folds if 0 <= f < cfg.num_folds]
    if not cfg.folds:
        raise ValueError("cfg.folds is empty after validation")
    # `aug` accepts a bool / None / mapping: normalise to a mapping.
    if cfg.aug is None:
        cfg.aug = {}
    elif isinstance(cfg.aug, bool):
        cfg.aug = {"enabled": bool(cfg.aug)}
    elif not isinstance(cfg.aug, dict):
        raise ValueError(f"cfg.aug must be a mapping, bool or null, got {type(cfg.aug).__name__}")
    return cfg


_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "filelock",
    "matplotlib",
    "PIL",
    "numba",
)


def setup_logging(level: int = logging.INFO, quiet_libs: bool = True) -> None:
    """Configure console logging for the CLI entry points.

    Silences the noisy hub/HTTP loggers and the HF progress bars so training
    output stays readable (``--quiet`` still works via the ``level`` argument).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if not quiet_libs:
        return
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    try:  # optional: transformers may not be installed in feature-only scripts
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_warning()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover - transformers absent
        pass


def _read_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(raw).__name__}")
    return dict(raw)


def load_config(
    path: Optional[str | os.PathLike[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML (+ optional overrides).

    Parameters
    ----------
    path:
        YAML file. May contain an ``extends: <other.yaml>`` key to inherit from
        another config (useful for the Kaggle ``data_root`` override written by
        :mod:`src.kaggle_setup`).
    overrides:
        Mapping merged on top of the file contents (e.g. CLI flags).
    strict:
        When True, unknown config keys raise instead of being ignored.
    """
    raw: Dict[str, Any] = {}
    if path is not None:
        raw = _read_yaml(path)
        parent = raw.pop("extends", None)
        if parent:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = Path(path).resolve().parent / parent_path
            raw = {**_read_yaml(parent_path), **raw}

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    fields = {f.name: f.default for f in dataclasses.fields(ExperimentConfig)}
    known = set(fields)
    unknown = set(raw) - known
    if unknown:
        msg = f"Unknown config keys (ignored): {sorted(unknown)}"
        if strict:
            raise ValueError(msg)
        LOG.warning(msg)

    cleaned = _coerce_scalars({k: v for k, v in raw.items() if k in known}, fields)
    cfg = ExperimentConfig(**{**{k: v for k, v in cleaned.items()}})
    return _validate(cfg)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the resolved DataVerse SER experiment config."
    )
    parser.add_argument("--config", default=None, help="YAML config path (optional).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of YAML.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Ad-hoc overrides, e.g. --set batch_size=4 epochs=2",
    )
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)

    setup_logging()
    cfg = load_config(args.config, overrides)
    print(json.dumps(cfg.to_dict(), indent=2, default=str) if args.json else cfg.to_yaml())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
