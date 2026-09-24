"""Fine-tuning CLI for speech SSL backbones (manual training loop, no HF Trainer).

Examples
--------
Local (RTX 3050 6 GB)::

    python -m src.train_ssl --config configs/base_superb.yaml --folds 0 1 2 3 4 --out runs/superb

Kaggle::

    python -m src.train_ssl --config configs/xlsr.kaggle.yaml --out /kaggle/working/runs/xlsr

Smoke test on CPU (requires network to download the checkpoint)::

    python -m src.train_ssl --config configs/base_superb.yaml --folds 0 --epochs 1 --dry-run

Features: AdamW with layer-wise LR decay, cosine/linear/onecycle schedules,
AMP, gradient accumulation, label smoothing, optional mixup, early stopping on
fold val macro-F1, per-fold ``best.pt`` + OOF artefacts, deterministic seeding.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMOTIONS, IDX_TO_EMOTION, ExperimentConfig, load_config, setup_logging
from .data import load_aux_df, read_train_df, split_summary
from .dataset import SERDataset, make_dataloader
from .eval import macro_f1
from .modeling import build_model, load_state_dict_flexible, prepare_inputs
from .splits import fold_assignment_path, fold_split, load_or_create_folds, save_folds

def _parse_set_args(items, parser) -> Dict[str, Any]:
    """Parse ``--set key=value`` items into a config override mapping."""
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


LOG = logging.getLogger("train_ssl")

__all__ = ["main", "set_seed", "soft_cross_entropy", "build_optimizer", "build_scheduler", "train_one_epoch", "evaluate"]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (and CUDA).

    ``torch.use_deterministic_algorithms`` is deliberately NOT enabled: several
    wav2vec2 kernels are non-deterministic and would raise instead.
    """
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy accepting int labels or soft (one-hot/mixup) targets."""
    num_classes = logits.size(-1)
    if targets.dim() == 1:
        return F.cross_entropy(logits, targets.long(), label_smoothing=float(label_smoothing))
    soft = targets.float()
    if label_smoothing > 0:
        soft = soft * (1.0 - label_smoothing) + label_smoothing / num_classes
    return -(soft * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def make_grad_scaler(enabled: bool):
    """``torch.amp.GradScaler`` with a fallback for older torch builds."""
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))  # type: ignore[attr-defined]


def autocast_context(device: torch.device, precision: str):
    """Return a (possibly null) autocast context manager."""
    if device.type != "cuda" or precision == "none":
        from contextlib import nullcontext

        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


# --------------------------------------------------------------------------- #
# Optimizer / scheduler
# --------------------------------------------------------------------------- #
_NO_DECAY_TOKENS = ("bias", "LayerNorm", "layer_norm", "layernorm", "ln_", "norm")


def _split_weight_decay(params: Iterable[nn.Parameter], name: str, weight_decay: float):
    no_decay = any(tok in name for tok in _NO_DECAY_TOKENS)
    return {"params": [p for p in params if p.requires_grad], "weight_decay": 0.0 if no_decay else float(weight_decay)}


def _encoder_layer_index(name: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.AdamW:
    """AdamW with layer-wise LR decay (head >> upper encoder >> lower encoder)."""
    head_lr = float(cfg.head_lr) if float(cfg.head_lr) > 0 else float(cfg.lr)
    encoder_lr = float(cfg.lr) * float(cfg.encoder_lr_scale)

    head_params = model.head_parameters()
    groups: List[Dict[str, Any]] = []
    if head_params:
        groups.append({"name": "head", **_split_weight_decay(head_params, "head", cfg.weight_decay), "lr": head_lr})

    encoder = model.backbone.encoder if hasattr(model.backbone, "encoder") else model.backbone
    n_layers = len(getattr(encoder, "layers", []) or [])
    by_index: Dict[Optional[int], List[Tuple[str, nn.Parameter]]] = {}
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        idx = _encoder_layer_index(name)
        by_index.setdefault(idx if idx is not None else None, []).append((name, param))

    top_index = max([i for i in by_index if i is not None], default=-1)
    for idx in sorted(by_index, key=lambda i: (i is None, i if i is not None else -1), reverse=False):
        entries = by_index[idx]
        if idx is None:
            lr = encoder_lr
            group_name = "encoder_other"
        else:
            # deeper layers (higher index) keep the base encoder LR
            decay = float(cfg.layer_lr_decay) ** max(top_index - idx, 0)
            lr = encoder_lr * decay
            group_name = f"encoder_layer_{idx}"
        params = [p for _, p in entries]
        names = " ".join(n for n, _ in entries)
        groups.append({"name": group_name, **_split_weight_decay(params, names, cfg.weight_decay), "lr": lr})

    if n_layers:
        LOG.info(
            "Optimizer: head_lr=%.2e, encoder_lr=%.2e (scale=%.2f, layer_decay=%.2f over %d layers)",
            head_lr,
            encoder_lr,
            cfg.encoder_lr_scale,
            cfg.layer_lr_decay,
            n_layers,
        )
    optimizer = torch.optim.AdamW(groups, lr=head_lr, betas=(0.9, 0.98), eps=1e-8)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    steps_per_epoch: int,
    epochs: int,
):
    """Warmup + cosine / linear / constant schedule (or OneCycle when asked)."""
    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    warmup_steps = int(round(float(cfg.warmup_frac) * total_steps))
    name = str(cfg.scheduler).lower()
    if name in ("cosine", "linear", "constant"):
        from transformers import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup,
        )

        factory = {
            "cosine": get_cosine_schedule_with_warmup,
            "linear": get_linear_schedule_with_warmup,
            "constant": get_constant_schedule_with_warmup,
        }[name]
        return factory(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if name == "onecycle":
        max_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=max(float(cfg.warmup_frac), 1e-3),
            anneal_strategy="cos",
        )
    return None


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def _mixup_prob(cfg: ExperimentConfig) -> float:
    aug = cfg.aug if isinstance(cfg.aug, dict) else {}
    mixup_cfg = aug.get("mixup") if isinstance(aug.get("mixup"), dict) else {}
    return float(mixup_cfg.get("prob", 1.0))


def train_one_epoch(
    model: nn.Module,
    feature_extractor: Any,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    cfg: ExperimentConfig,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """One full epoch with gradient accumulation + AMP + optional mixup."""
    from .augment import mixup as mixup_fn

    model.train()
    precision = cfg.precision
    use_amp = device.type == "cuda" and precision != "none"
    accum = max(1, int(cfg.grad_accum))
    prob_mixup = _mixup_prob(cfg)

    running_loss, seen = 0.0, 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    t0 = time.time()

    n_batches = len(loader) if hasattr(loader, "__len__") else None  # type: ignore[arg-type]
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(loader, desc=f"epoch {epoch} train", leave=False, total=n_batches, disable=None)
    except Exception:  # pragma: no cover - tqdm absent
        iterator = loader

    optimizer.zero_grad(set_to_none=True)
    for step, (waves, labels, _lengths) in enumerate(iterator):
        if max_batches is not None and step >= int(max_batches):
            break
        waves = waves.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        soft_targets: Optional[torch.Tensor] = None
        if cfg.mixup_alpha and cfg.mixup_alpha > 0 and np.random.rand() < prob_mixup:
            waves, soft_targets = mixup_fn(
                waves, labels, alpha=float(cfg.mixup_alpha), num_classes=model.num_labels
            )

        inputs = prepare_inputs(waves, feature_extractor, cfg.sample_rate)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with autocast_context(device, precision):
            logits = model(**inputs)
            targets = soft_targets if soft_targets is not None else labels
            loss = soft_cross_entropy(logits, targets, cfg.label_smoothing)

        loss_scaled = loss / accum
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_step = ((step + 1) % accum == 0) or (step + 1 == n_batches) or (
            max_batches is not None and step + 1 >= int(max_batches)
        )
        if is_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(cfg.grad_clip)
                )
            stepped = True
            if scaler is not None and scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # FP16 overflow -> the optimizer step was skipped; don't advance
                # the LR schedule in that case (matches HF Trainer behaviour).
                stepped = scaler.get_scale() >= scale_before
                if not stepped:
                    LOG.debug("GradScaler skipped an optimizer step (fp16 overflow)")
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and stepped:
                scheduler.step()

        running_loss += float(loss.detach().item()) * batch_size
        seen += batch_size
        all_preds.append(logits.detach().argmax(dim=-1).float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

    metrics: Dict[str, float] = {"loss": running_loss / max(seen, 1), "seconds": time.time() - t0, "n": seen}
    if all_labels:
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds).astype(int)
        # NOTE: with mixup on, this "train F1" is against the un-mixed labels
        # (still a useful overfitting signal).
        metrics["macro_f1"] = macro_f1(y_true, y_pred)
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    feature_extractor: Any,
    loader: Iterable,
    cfg: ExperimentConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(probs[N, C], labels[N], mean_loss)`` in loader order."""
    model.eval()
    precision = cfg.precision
    probs: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []
    losses, seen = 0.0, 0

    for step, (waves, labels, _lengths) in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        waves = waves.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = prepare_inputs(waves, feature_extractor, cfg.sample_rate)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with autocast_context(device, precision):
            logits = model(**inputs)
        loss = soft_cross_entropy(logits, labels, 0.0)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        labels_out.append(labels.cpu().numpy())
        losses += float(loss.item()) * labels.size(0)
        seen += labels.size(0)

    if not probs:  # pragma: no cover - empty loader
        return np.zeros((0, model.num_labels), dtype=np.float32), np.zeros((0,), dtype=np.int64), float("nan")
    return (
        np.concatenate(probs, axis=0).astype(np.float32),
        np.concatenate(labels_out, axis=0).astype(np.int64),
        losses / max(seen, 1),
    )


# --------------------------------------------------------------------------- #
# Fold artefacts
# --------------------------------------------------------------------------- #
def save_fold_outputs(
    fold_dir: str,
    filenames: Sequence[str],
    fold_ids: Sequence[int],
    probs: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Write ``oof_probs.npy`` + ``oof_preds.csv`` for one fold."""
    os.makedirs(fold_dir, exist_ok=True)
    probs = np.asarray(probs, dtype=np.float32)
    n = len(probs)
    if len(filenames) != n or len(labels) != n:
        LOG.warning(
            "OOF length mismatch (filenames=%d, labels=%d, probs=%d); keeping the first %d rows.",
            len(filenames),
            len(labels),
            n,
            min(n, len(filenames), len(labels)),
        )
        n = min(n, len(filenames), len(labels))
        probs = probs[:n]
    filenames = list(filenames)[:n]
    fold_ids = list(fold_ids)[:n]
    labels = np.asarray(labels)[:n]

    np.save(os.path.join(fold_dir, "oof_probs.npy"), probs)
    preds = probs.argmax(axis=1)
    csv_path = os.path.join(fold_dir, "oof_preds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "fold", "true", "pred"] + [f"prob_{e}" for e in EMOTIONS])
        for i, name in enumerate(filenames):
            writer.writerow(
                [name, int(fold_ids[i]), int(labels[i]), int(preds[i])]
                + [f"{float(p):.6f}" for p in probs[i]]
            )
    LOG.info("Wrote %s and oof_probs.npy (%d rows)", csv_path, len(filenames))


def write_train_log(fold_dir: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = os.path.join(fold_dir, "train_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("Wrote %s", path)


# --------------------------------------------------------------------------- #
# Fold driver
# --------------------------------------------------------------------------- #
def run_fold(
    cfg: ExperimentConfig,
    train_df,
    fold: int,
    device: torch.device,
    out_root: str,
    aux_df=None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Train one fold, saving ``best.pt`` / OOF artefacts; returns best metrics."""
    fold_dir = os.path.join(out_root, f"fold{int(fold)}")
    os.makedirs(fold_dir, exist_ok=True)
    set_seed(int(cfg.seed) + int(fold))

    train_part, val_part = fold_split(train_df, fold)
    if aux_df is not None and len(aux_df) > 0:
        # external rows (fold=-1) are added to TRAIN only; val/OOF untouched
        train_part = pd.concat([train_part.reset_index(drop=True), aux_df.reset_index(drop=True)], ignore_index=True)
        LOG.info("fold %d | train=%d (incl. %d aux) val=%d", fold, len(train_part), len(aux_df), len(val_part))
    train_ds = SERDataset(train_part, cfg.data_root, cfg, fold=fold, split="train", augment=True)
    val_ds = SERDataset(val_part, cfg.data_root, cfg, fold=fold, split="val", augment=False)
    train_loader = make_dataloader(train_ds, cfg, shuffle=True, drop_last=False)
    val_loader = make_dataloader(val_ds, cfg, shuffle=False)

    model, feature_extractor = build_model(cfg, num_labels=len(EMOTIONS), device=str(device))

    steps_per_epoch = math.ceil(len(train_loader) / max(1, int(cfg.grad_accum)))
    if max_train_batches is not None:
        steps_per_epoch = math.ceil(min(len(train_loader), int(max_train_batches)) / max(1, int(cfg.grad_accum)))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs)
    scaler = make_grad_scaler(
        enabled=(device.type == "cuda" and cfg.precision == "fp16")
    )
    history: List[Dict[str, Any]] = []
    best_path = os.path.join(fold_dir, "best.pt")

    LOG.info(
        "fold %d | train=%d val=%d | steps/epoch=%d | %s",
        fold,
        len(train_ds),
        len(val_ds),
        steps_per_epoch,
        "cuda" if device.type == "cuda" else "cpu",
    )
    try:
        return _train_fold_loop(
            cfg, train_ds, val_ds, train_loader, val_loader, model, feature_extractor,
            optimizer, scheduler, scaler, fold, fold_dir, best_path, device,
            max_train_batches, max_val_batches, history,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            LOG.error(
                "CUDA OOM while training fold %d. Try: --set batch_size=2 grad_accum=8 "
                "(or 1/16) and keep grad_ckpt: true, lower max_seconds to 4.0, and make sure "
                "no other process is holding GPU memory (this box is shared). "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is already set by this CLI.",
                fold,
            )
        raise


def _train_fold_loop(
    cfg: ExperimentConfig,
    train_ds: SERDataset,
    val_ds: SERDataset,
    train_loader: Any,
    val_loader: Any,
    model: nn.Module,
    feature_extractor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    fold: int,
    fold_dir: str,
    best_path: str,
    device: torch.device,
    max_train_batches: Optional[int],
    max_val_batches: Optional[int],
    history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Inner epoch loop of :func:`run_fold` (separated for OOM handling)."""
    best_f1, best_epoch, patience_left = -1.0, -1, int(cfg.early_stop_patience)
    for epoch in range(1, int(cfg.epochs) + 1):
        train_ds.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, feature_extractor, train_loader, optimizer, scheduler, scaler,
            cfg, device, epoch=epoch, max_batches=max_train_batches,
        )
        val_probs, val_labels, val_loss = evaluate(
            model, feature_extractor, val_loader, cfg, device, max_batches=max_val_batches
        )
        val_pred = val_probs.argmax(axis=1)
        val_f1 = macro_f1(val_labels, val_pred)
        current_lr = float(optimizer.param_groups[0]["lr"])
        LOG.info(
            "fold %d | epoch %d/%d | train_loss %.4f | val_loss %.4f | val_macro_f1 %.4f | lr %.2e | %.1fs",
            fold, epoch, cfg.epochs, train_metrics["loss"], val_loss, val_f1, current_lr,
            train_metrics["seconds"],
        )
        history.append(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_macro_f1": round(train_metrics.get("macro_f1", float("nan")), 6),
                "val_loss": round(val_loss, 6),
                "val_macro_f1": round(val_f1, 6),
                "lr": current_lr,
                "seconds": round(train_metrics["seconds"], 2),
            }
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1, best_epoch = val_f1, epoch
            patience_left = int(cfg.early_stop_patience)
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if cfg.early_stop_patience > 0 and patience_left <= 0:
                LOG.info("fold %d | early stopping at epoch %d (best %.4f @ %d)", fold, epoch, best_f1, best_epoch)
                break

    write_train_log(fold_dir, history)

    # Reload the best epoch and regenerate its validation probabilities.
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        load_state_dict_flexible(model, state, strict=False)
    val_probs, val_labels, val_loss = evaluate(
        model, feature_extractor, val_loader, cfg, device, max_batches=max_val_batches
    )
    val_f1 = macro_f1(val_labels, val_probs.argmax(axis=1))
    LOG.info("fold %d | best val macro-F1 %.4f (epoch %d)", fold, val_f1, best_epoch)

    filenames, fold_ids = list(val_ds.filenames), list(val_ds.folds)
    if max_val_batches is not None:
        LOG.warning(
            "--limit-val-batches=%s: OOF artefacts cover only the first %d/%d validation rows.",
            max_val_batches,
            len(val_labels),
            len(val_ds),
        )
        filenames, fold_ids = filenames[: len(val_labels)], fold_ids[: len(val_labels)]

    save_fold_outputs(
        fold_dir,
        filenames=filenames,
        fold_ids=fold_ids,
        probs=val_probs,
        labels=val_labels,
    )
    with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "val_macro_f1_reloaded": val_f1,
                "val_loss": val_loss,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
            fh,
            indent=2,
        )
        fh.write("\n")
    return {"fold": fold, "best_val_macro_f1": best_f1, "best_epoch": best_epoch, "n_val": len(val_ds)}


# --------------------------------------------------------------------------- #
# Dry run (no data needed)
# --------------------------------------------------------------------------- #
def dry_run(cfg: ExperimentConfig, device: torch.device) -> int:
    """Build the model and run one forward/backward on a synthetic batch."""
    set_seed(cfg.seed)
    LOG.info("Dry run: building %s on %s ...", cfg.model_name, device)
    model, fe = build_model(cfg, num_labels=len(EMOTIONS), device=str(device))
    n_samples = cfg.max_samples
    waves = torch.randn(2, 1, n_samples)
    labels = torch.tensor([0, 3], dtype=torch.long, device=device)
    inputs = prepare_inputs(waves, fe, cfg.sample_rate)
    LOG.info("Feature extractor output keys: %s", {k: tuple(v.shape) for k, v in inputs.items()})
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.train()
    with autocast_context(device, cfg.precision):
        logits = model(**inputs)
        loss = soft_cross_entropy(logits, labels, cfg.label_smoothing)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    LOG.info("Dry run OK: logits %s, loss %.4f, params with grads: %d", tuple(logits.shape), float(loss.detach()), n_grad)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train_ssl",
        description="Fine-tune a speech SSL model for 4-class SER (macro-F1).",
    )
    parser.add_argument("--config", default=None, help="YAML config (default: built-in defaults).")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Folds to run (default: cfg.folds).")
    parser.add_argument("--out", default=None, help="Output root, e.g. runs/xlsr (default: cfg.output_dir/<tag>).")
    parser.add_argument("--tag", default=None, help="Run tag (default: model name slug).")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cfg.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers.")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu | cuda:1 ...")
    parser.add_argument("--no-aug", action="store_true", help="Disable all waveform augmentation.")
    parser.add_argument(
        "--folds-json", default=None, help="Fold manifest to reuse/create (default: <data_root>/folds.json)."
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Config overrides, e.g. --set data_root=data/SRCASW-BhavVaani batch_size=4",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build model + 1 fwd/bwd on synthetic audio, then exit.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Debug: cap train batches/epoch.")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="Debug: cap val batches.")
    parser.add_argument("--quiet", action="store_true", help="Only warnings on the console.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.WARNING if args.quiet else logging.INFO)

    overrides: Dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    overrides.update(_parse_set_args(args.set, build_parser()))
    cfg = load_config(args.config, overrides)
    if args.no_aug:
        cfg.aug = {k: False for k in cfg.aug}

    # Reduce CUDA fragmentation for long (6 s) sequences on small cards; must be
    # set before the caching allocator initialises. No-op if the user set it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = resolve_device(args.device)
    if device.type == "cuda":
        LOG.info("CUDA device: %s", torch.cuda.get_device_name(device))

    if args.dry_run:
        return dry_run(cfg, device)

    out_root = args.out or cfg.run_dir(args.tag)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(cfg.to_yaml())

    train_df = read_train_df(cfg.data_root, cfg)
    LOG.info(split_summary(train_df))
    folds_path = args.folds_json or fold_assignment_path(cfg.data_root, out_root)
    train_df = load_or_create_folds(train_df, cfg.num_folds, cfg.seed, path=folds_path)
    LOG.info("Fold sizes: %s", train_df["fold"].value_counts().sort_index().to_dict())
    # Copy the manifest next to the run so predict.py uses exactly these folds.
    try:
        save_folds(
            dict(zip(train_df["filename"].astype(str), train_df["fold"])),
            os.path.join(out_root, "folds.json"),
            n_folds=cfg.num_folds,
            seed=cfg.seed,
        )
    except OSError as exc:  # pragma: no cover - read-only output dir
        LOG.warning("Could not copy the fold manifest into %s: %s", out_root, exc)
        # fall back to a local copy inside the output dir
        try:
            save_folds(
                dict(zip(train_df["filename"].astype(str), train_df["fold"])),
                fold_assignment_path(cfg.data_root, out_root),
                n_folds=cfg.num_folds,
                seed=cfg.seed,
            )
        except OSError:
            pass

    folds = args.folds if args.folds else list(cfg.folds)
    LOG.info("Training %s -> %s", cfg.model_name, out_root)

    aux_df = load_aux_df(cfg)
    if aux_df is not None:
        LOG.info("Aux (external) training data: %d rows -> train splits only", len(aux_df))

    results = []
    for fold in folds:
        results.append(
            run_fold(
                cfg,
                train_df,
                fold,
                device,
                out_root,
                aux_df=aux_df,
                max_train_batches=args.limit_train_batches,
                max_val_batches=args.limit_val_batches,
            )
        )

    summary_path = os.path.join(out_root, "cv_summary.json")
    f1s = [r["best_val_macro_f1"] for r in results]
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": cfg.model_name,
                "folds": results,
                "mean_best_val_macro_f1": float(np.mean(f1s)) if f1s else None,
                "std_best_val_macro_f1": float(np.std(f1s)) if f1s else None,
            },
            fh,
            indent=2,
        )
    LOG.info("CV mean macro-F1: %.4f (+/- %.4f) -> %s", float(np.mean(f1s)), float(np.std(f1s)), summary_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
