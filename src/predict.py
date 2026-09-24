"""Inference CLI: OOF + test probabilities for a trained run.

Examples
--------
::

    python -m src.predict --config configs/base_superb.yaml --ckpt runs/superb --out oof/superb
    python -m src.predict --ckpt runs/superb --out oof/superb --tta       # 3-crop TTA

Outputs (``--out`` directory)::

    oof_probs_<tag>.csv   filename,fold,true,prob_angry,prob_happy,prob_neutral,prob_sad
    oof_probs_<tag>.npy   float32 (N_train, 4)
    test_probs_<tag>.csv  filename,prob_angry,prob_happy,prob_neutral,prob_sad
    test_probs_<tag>.npy  float32 (N_test, 4)
    oof_probs.csv / test_probs.csv   untagged copies (convenience)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from .config import EMOTIONS, ExperimentConfig, load_config, setup_logging
from .data import load_audio_raw, read_test_df, read_train_df, resolve_audio_path
from .dataset import SERDataset, get_cache
from .eval import macro_f1
from .modeling import build_model, load_state_dict_flexible, prepare_inputs
from .splits import fold_assignment_path, load_or_create_folds

def _parse_set_args(items, parser) -> Dict[str, Any]:
    """Parse ``--set key=value`` items into a config override mapping."""
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


LOG = logging.getLogger("predict")


# --------------------------------------------------------------------------- #
# Crop generation
# --------------------------------------------------------------------------- #
def _crop_starts(total: int, want: int, rng: np.random.Generator) -> List[int]:
    if total <= want:
        return [0]
    return sorted(set(int(s) for s in rng.integers(0, total - want + 1, size=max(1, 8))))


def make_crops(
    wave: np.ndarray,
    n_samples: int,
    n_crops: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Deterministic centre crop + random crops (TTA variants)."""
    crops: List[np.ndarray] = []
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size <= n_samples:
        return [np.pad(wave, (0, max(0, n_samples - wave.size)))[:n_samples].astype(np.float32)]

    center = (wave.size - n_samples) // 2
    starts = [center] + _crop_starts(wave.size, n_samples, rng)[: max(0, int(n_crops) - 1)]
    for start in starts:
        if len(crops) >= int(n_crops):
            break
        crops.append(wave[start : start + n_samples].astype(np.float32))
    while len(crops) < int(n_crops):  # pad with the centre crop if too few
        crops.append(crops[0])
    return crops


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_rows(
    model: torch.nn.Module,
    feature_extractor: Any,
    df,
    cfg: ExperimentConfig,
    device: torch.device,
    n_crops: int = 1,
    batch_size: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Average-crop probabilities for every row of ``df`` -> ``(N, 4)``."""
    model.eval()
    n_samples = int(cfg.max_samples)
    batch_size = int(batch_size or cfg.batch_size)
    rng = np.random.default_rng(int(seed))
    cache = get_cache()
    out: List[np.ndarray] = []

    paths = df["path"].tolist() if "path" in df.columns else [None] * len(df)
    filenames = [str(f) for f in df["filename"].tolist()]

    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        batch_waves: List[np.ndarray] = []
        counts: List[int] = []
        for i in range(start, stop):
            path = paths[i] or resolve_audio_path(cfg.data_root, filenames[i], split="test", cfg=cfg)
            wave = cache.get(str(path))
            if wave is None:
                if path is None:
                    LOG.warning("Missing audio for %s - using silence", filenames[i])
                    wave = np.zeros(n_samples, dtype=np.float32)
                else:
                    wave = load_audio_raw(
                        path, sr=cfg.sample_rate, trim=cfg.trim_silence, top_db=cfg.trim_top_db
                    )
                cache[str(path)] = wave
            crops = make_crops(wave, n_samples, max(1, int(n_crops)), rng)
            batch_waves.extend(crops)
            counts.append(len(crops))

        waves = torch.from_numpy(np.stack(batch_waves)).unsqueeze(1)  # (sum(counts), 1, T)
        inputs = {k: v.to(device) for k, v in prepare_inputs(waves, feature_extractor, cfg.sample_rate).items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda" and cfg.precision == "fp16")):
            logits = model(**inputs)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        # Average the crops belonging to each file (the crop count can be < n_crops
        # for clips shorter than max_seconds).
        offset = 0
        for count in counts:
            out.append(probs[offset : offset + count].mean(axis=0, keepdims=True))
            offset += count
    if not out:
        return np.zeros((0, len(EMOTIONS)), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_test_probs(path: str, filenames: Sequence[str], probs: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename"] + [f"prob_{e}" for e in EMOTIONS])
        for name, row in zip(filenames, probs):
            writer.writerow([name] + [f"{float(p):.6f}" for p in row])


def write_oof_probs(
    path: str,
    filenames: Sequence[str],
    folds: Sequence[int],
    truths: Sequence[int],
    probs: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "fold", "true"] + [f"prob_{e}" for e in EMOTIONS])
        for name, fold, true, row in zip(filenames, folds, truths, probs):
            writer.writerow([name, int(fold), int(true)] + [f"{float(p):.6f}" for p in row])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Predict OOF + test probabilities from trained fold checkpoints.",
    )
    parser.add_argument("--config", default=None, help="YAML config (default: <ckpt>/config.yaml).")
    parser.add_argument("--ckpt", required=True, help="Run directory containing fold*/best.pt.")
    parser.add_argument("--out", default=None, help="Output directory (default: <cfg.oof_dir>/<tag>).")
    parser.add_argument("--tag", default=None, help="Artefact tag (default: basename of --ckpt).")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Folds to predict (default: all found).")
    parser.add_argument("--tta", dest="tta", action="store_true", default=None, help="Force TTA on.")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Force TTA off.")
    parser.add_argument("--tta-crops", type=int, default=None, help="Number of crops per file (default cfg.tta_crops).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cfg.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers.")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--seed", type=int, default=None, help="TTA RNG seed (default cfg.seed).")
    parser.add_argument(
        "--folds-json", default=None, help="Fold manifest (default: <ckpt>/folds.json, else <data_root>/folds.json)."
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Config overrides, e.g. --set data_root=data/SRCASW-BhavVaani",
    )
    return parser


def load_config_for_ckpt(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)
    fallback = os.path.join(args.ckpt, "config.yaml")
    if os.path.exists(fallback):
        LOG.info("No --config given; using %s", fallback)
        return load_config(fallback)
    LOG.warning("No --config and no %s; using built-in defaults.", fallback)
    return load_config(None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if not os.path.isdir(args.ckpt):
        LOG.error("--ckpt directory not found: %s", args.ckpt)
        return 2

    cfg = load_config_for_ckpt(args).apply_overrides(_parse_set_args(args.set, build_parser()))
    if args.batch_size:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    use_tta = bool(cfg.tta if args.tta is None else args.tta)
    n_crops = int(args.tta_crops or cfg.tta_crops) if use_tta else 1
    tag = args.tag or os.path.basename(os.path.normpath(args.ckpt))
    out_dir = args.out or os.path.join(cfg.oof_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    LOG.info("Predicting %s | tag=%s | TTA=%s (%d crop(s)) | device=%s", cfg.model_name, tag, use_tta, n_crops, device)

    # ----- data ---------------------------------------------------------- #
    train_df = read_train_df(cfg.data_root, cfg)
    test_df = read_test_df(cfg.data_root, cfg)
    folds_path = args.folds_json or os.path.join(args.ckpt, "folds.json")
    if not os.path.exists(folds_path):
        folds_path = fold_assignment_path(cfg.data_root, args.ckpt)
    train_df = load_or_create_folds(train_df, cfg.num_folds, cfg.seed, path=folds_path)

    # ----- discover folds ------------------------------------------------- #
    fold_dirs = sorted(
        [
            d for d in os.listdir(args.ckpt)
            if d.startswith("fold") and os.path.exists(os.path.join(args.ckpt, d, "best.pt"))
        ],
        key=lambda s: int("".join(c for c in s if c.isdigit()) or 0),
    )
    if args.folds:
        fold_dirs = [d for d in fold_dirs if int("".join(c for c in d if c.isdigit()) or -1) in set(args.folds)]
    if not fold_dirs:
        LOG.error("No fold*/best.pt checkpoints found under %s", args.ckpt)
        return 2

    # ----- per-fold inference -------------------------------------------- #
    model, feature_extractor = build_model(cfg, num_labels=len(EMOTIONS), device=str(device))
    oof_parts: List[Tuple[List[str], List[int], List[int], np.ndarray]] = []
    test_probs_list: List[np.ndarray] = []
    fold_scores: Dict[str, float] = {}

    for fold_name in fold_dirs:
        fold = int("".join(c for c in fold_name if c.isdigit()) or 0)
        ckpt_path = os.path.join(args.ckpt, fold_name, "best.pt")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_state_dict_flexible(model, state, strict=False)
        LOG.info("Loaded %s", ckpt_path)

        # OOF rows for this fold
        val_rows = train_df.loc[train_df["fold"] == fold].reset_index(drop=True)
        val_probs = predict_rows(
            model, feature_extractor, val_rows, cfg, device,
            n_crops=n_crops, batch_size=cfg.batch_size, seed=cfg.seed + fold,
        )
        val_true = [int(v) for v in val_rows["emotion"].map({e: i for i, e in enumerate(EMOTIONS)})]
        oof_parts.append(([str(f) for f in val_rows["filename"]], [fold] * len(val_rows), val_true, val_probs))
        fold_scores[fold_name] = macro_f1(val_true, val_probs.argmax(axis=1))
        LOG.info("%s | val macro-F1 %.4f (%d rows)", fold_name, fold_scores[fold_name], len(val_rows))

        if not test_df.empty:
            test_probs_list.append(
                predict_rows(
                    model, feature_extractor, test_df, cfg, device,
                    n_crops=n_crops, batch_size=cfg.batch_size, seed=cfg.seed + fold,
                )
            )

    # ----- save ----------------------------------------------------------- #
    filenames = [n for part in oof_parts for n in part[0]]
    folds_all = [f for part in oof_parts for f in part[1]]
    truths = [t for part in oof_parts for t in part[2]]
    oof_probs = np.concatenate([part[3] for part in oof_parts], axis=0)

    oof_csv = os.path.join(out_dir, f"oof_probs_{tag}.csv")
    write_oof_probs(oof_csv, filenames, folds_all, truths, oof_probs)
    np.save(os.path.join(out_dir, f"oof_probs_{tag}.npy"), oof_probs)
    write_oof_probs(os.path.join(out_dir, "oof_probs.csv"), filenames, folds_all, truths, oof_probs)
    LOG.info("OOF macro-F1 (all folds): %.4f -> %s", macro_f1(truths, oof_probs.argmax(axis=1)), oof_csv)

    test_csv_path = None
    if test_probs_list:
        test_probs = np.mean(np.stack(test_probs_list, axis=0), axis=0).astype(np.float32)
        test_csv_path = os.path.join(out_dir, f"test_probs_{tag}.csv")
        write_test_probs(test_csv_path, [str(f) for f in test_df["filename"]], test_probs)
        np.save(os.path.join(out_dir, f"test_probs_{tag}.npy"), test_probs)
        write_test_probs(os.path.join(out_dir, "test_probs.csv"), [str(f) for f in test_df["filename"]], test_probs)
        LOG.info("Test probs -> %s", test_csv_path)

    with open(os.path.join(out_dir, f"predict_meta_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tag": tag,
                "ckpt": args.ckpt,
                "model_name": cfg.model_name,
                "tta": use_tta,
                "tta_crops": n_crops,
                "fold_macro_f1": fold_scores,
                "oof_macro_f1": float(macro_f1(truths, oof_probs.argmax(axis=1))),
                "nfolds_used": len(fold_dirs),
                "n_train": int(len(filenames)),
                "n_test": int(len(test_df)),
            },
            fh,
            indent=2,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
