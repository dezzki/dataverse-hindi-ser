"""Ensemble OOF probability files: weights + temperature + class priors.

Pipeline
--------
1. Load ``oof_probs_<tag>.csv`` (+ ``test_probs_<tag>.csv``) for every tag.
2. Optionally fit a per-model temperature on OOF log-loss.
3. Optimise non-negative simplex weights (softmax parameterisation) with
   ``scipy.optimize.minimize`` (Nelder-Mead, several restarts) + a greedy
   polish, maximising the macro-F1 of the blended OOF probabilities.
4. Apply the prior/bias + smoothing search from :mod:`src.eval`.
5. Write ``test_probs.csv`` (filename + 4 probs), ``weights.json`` and
   ``score_report.txt`` into ``--out``.

Usage
-----
::

    python -m src.ensemble --tags superb xlsr whisper --out runs/final
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import EMOTIONS, setup_logging
from .eval import (
    apply_calibration,
    best_threshold_and_bias,
    fit_temperature,
    macro_f1,
    per_class_f1,
    temperature_scale,
)

LOG = logging.getLogger("ensemble")

_PROB_COLS = [f"prob_{e}" for e in EMOTIONS]


# --------------------------------------------------------------------------- #
# Loading / alignment
# --------------------------------------------------------------------------- #
def find_probs(tag: str, kind: str, root: str) -> Optional[str]:
    """Locate ``{root}/<tag>/{kind}_probs_<tag>.csv`` with sensible fallbacks.

    ``kind`` is ``oof`` or ``test``.
    """
    stem = f"{kind}_probs"
    candidates = [
        os.path.join(root, tag, f"{stem}_{tag}.csv"),
        os.path.join(root, f"{stem}_{tag}.csv"),
        os.path.join(root, tag, f"{stem}.csv"),
        os.path.join(root, f"{stem}.csv"),
    ]
    if os.path.isdir(tag):
        candidates += [os.path.join(tag, f"{stem}_{tag}.csv"), os.path.join(tag, f"{stem}.csv")]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    hits = sorted(glob.glob(os.path.join(root, "**", f"{stem}_{tag}.csv"), recursive=True))
    return hits[0] if hits else None


def read_probs(path: str) -> pd.DataFrame:
    """Read a probability CSV -> frame[filename, prob_* , (fold, true)]."""
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError(f"{path} must have a 'filename' column")
    missing = [c for c in _PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing probability columns {missing}")
    out = df[["filename"] + _PROB_COLS].copy()
    out["filename"] = out["filename"].astype(str)
    if out["filename"].duplicated().any():
        raise ValueError(f"{path} contains duplicated filenames")
    for extra in ("fold", "true", "pred"):
        if extra in df.columns:
            out[extra] = df[extra].to_numpy()
    return out


def align_to(frame: pd.DataFrame, order: Sequence[str], path: str, what: str) -> np.ndarray:
    """Return ``frame`` probabilities reordered to ``order`` (all must exist)."""
    lookup = {f: i for i, f in enumerate(frame["filename"].tolist())}
    missing = [f for f in order if f not in lookup]
    if missing:
        raise ValueError(
            f"{what} {path} is missing {len(missing)} row(s) present in the reference file "
            f"(e.g. {missing[:3]}); re-run predict for all tags on the same split."
        )
    matrix = frame[_PROB_COLS].to_numpy(dtype=np.float32)
    return np.stack([matrix[lookup[f]] for f in order])


# --------------------------------------------------------------------------- #
# Weight optimisation
# --------------------------------------------------------------------------- #
def _softmax_weights(raw: np.ndarray) -> np.ndarray:
    e = np.exp(np.asarray(raw, dtype=np.float64) - np.max(raw))
    return e / e.sum()


def blend_with(weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """``(K, N, C)`` + ``(K,)`` -> ``(N, C)``."""
    return np.tensordot(np.asarray(weights, dtype=np.float64), probs, axes=(0, 0)).astype(np.float32)


def optimise_weights(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    seed: int = 42,
    maxiter: int = 400,
) -> Tuple[np.ndarray, float]:
    """Non-negative simplex weights maximising OOF macro-F1 (Nelder-Mead + polish)."""
    from scipy.optimize import minimize

    n_models = int(oof_probs.shape[0])
    if n_models == 1:
        return np.ones(1), macro_f1(y_true, oof_probs[0].argmax(axis=1))

    def neg_score(raw: np.ndarray) -> float:
        return -macro_f1(y_true, blend_with(_softmax_weights(raw), oof_probs).argmax(axis=1))

    inits: List[np.ndarray] = [np.zeros(n_models)]
    for i in range(n_models):
        hot = np.full(n_models, -4.0)
        hot[i] = 4.0
        inits.append(hot)
    rng = np.random.default_rng(seed)
    inits += [rng.normal(0, 0.5, size=n_models) for _ in range(2)]

    best_w, best_score = _softmax_weights(inits[0]), -neg_score(inits[0])
    for init in inits:
        res = minimize(
            neg_score, init, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6},
        )
        score = -float(res.fun)
        if score > best_score + 1e-9:
            best_w, best_score = _softmax_weights(res.x), score

    # Greedy polish: macro-F1 is piecewise constant, so NM can stall on plateaus.
    improved = True
    while improved:
        improved = False
        for i in range(n_models):
            for factor in (0.8, 1.25, 0.5, 2.0):
                trial = best_w.copy()
                trial[i] *= factor
                if trial.sum() <= 0:
                    continue
                trial /= trial.sum()
                score = macro_f1(y_true, blend_with(trial, oof_probs).argmax(axis=1))
                if score > best_score + 1e-9:
                    best_w, best_score, improved = trial, score, True
    return best_w, float(best_score)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ensemble", description="Blend + calibrate OOF probabilities into a final ensemble."
    )
    parser.add_argument("--tags", nargs="*", default=[], help="Run tags to blend (order preserved).")
    parser.add_argument("--oof-dir", default="oof", help="Directory containing <tag>/oof_probs_<tag>.csv.")
    parser.add_argument("--files", nargs="*", default=[], help="Explicit OOF CSV paths (overrides --tags).")
    parser.add_argument("--test-files", nargs="*", default=[], help="Explicit test CSV paths (same order as --files).")
    parser.add_argument("--out", default="runs/final", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-temperature", action="store_true", help="Skip per-model temperature scaling.")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip the class-prior/bias search.")
    parser.add_argument("--equal-weights", action="store_true", help="Plain average instead of weight search.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    os.makedirs(args.out, exist_ok=True)

    # ----- resolve input paths ------------------------------------------- #
    oof_paths: List[str] = []
    test_paths: List[Optional[str]] = []
    tags: List[str] = []
    if args.files:
        oof_paths = list(args.files)
        tags = [
            os.path.basename(os.path.dirname(p)) or os.path.splitext(os.path.basename(p))[0]
            for p in oof_paths
        ]
        test_paths = list(args.test_files) + [None] * max(0, len(oof_paths) - len(args.test_files))
    else:
        if not args.tags:
            raise SystemExit("Provide --tags or --files")
        for tag in args.tags:
            oof = find_probs(tag, "oof", args.oof_dir)
            if oof is None:
                raise SystemExit(f"No OOF probabilities found for tag {tag!r} under {args.oof_dir}/")
            test = find_probs(tag, "test", args.oof_dir)
            if test is None:
                LOG.warning("No test probabilities found for tag %r (it can still be blended on OOF)", tag)
            oof_paths.append(oof)
            test_paths.append(test)
            tags.append(tag)

    # ----- canonical OOF row order (from the first tag) ------------------- #
    first = read_probs(oof_paths[0])
    order = first["filename"].tolist()
    y_true = first["true"].to_numpy().astype(int) if "true" in first.columns else None
    folds = first["fold"].to_numpy().astype(int) if "fold" in first.columns else None
    if y_true is None:
        LOG.warning("%s has no 'true' column: weight/calibration search is disabled", oof_paths[0])

    oof_stack = [align_to(first, order, oof_paths[0], "OOF")]
    for tag, path in zip(tags[1:], oof_paths[1:]):
        frame = read_probs(path)
        oof_stack.append(align_to(frame, order, path, "OOF"))
        LOG.info("Loaded OOF %-16s %s", tag, path)
    stack = np.stack(oof_stack, axis=0)  # (K, N, C)

    # ----- test matrices (aligned to their own reference order) ----------- #
    test_stack_parts: List[np.ndarray] = []
    test_order: Optional[List[str]] = None
    keep_test: List[int] = []
    for i, (tag, path) in enumerate(zip(tags, test_paths)):
        if not path:
            continue
        frame = read_probs(path)
        if test_order is None:
            test_order = frame["filename"].tolist()
        test_stack_parts.append(align_to(frame, test_order, path, "test"))
        keep_test.append(i)
    test_stack = np.stack(test_stack_parts, axis=0) if test_stack_parts else None

    # ----- temperature scaling ------------------------------------------- #
    temperatures = [1.0] * len(tags)
    scaled = stack.copy()
    if not args.no_temperature:
        for i in range(len(tags)):
            if y_true is None:
                break
            try:
                temperatures[i] = float(fit_temperature(stack[i], y_true))
            except Exception as exc:  # pragma: no cover - scipy defensive
                LOG.warning("Temperature fit failed for %s: %s", tags[i], exc)
            scaled[i] = temperature_scale(stack[i], temperatures[i])
        LOG.info("Temperatures: %s", [round(t, 3) for t in temperatures])

    # ----- weights -------------------------------------------------------- #
    if args.equal_weights or y_true is None:
        weights = np.ones(len(tags), dtype=np.float64) / len(tags)
        blend_score = (
            macro_f1(y_true, blend_with(weights, scaled).argmax(axis=1)) if y_true is not None else float("nan")
        )
    else:
        weights, blend_score = optimise_weights(scaled, y_true, seed=args.seed)
    LOG.info("Weights: %s | raw blend OOF macro-F1 %.4f", [round(float(w), 4) for w in weights], blend_score)

    # ----- calibration ---------------------------------------------------- #
    blend_oof = blend_with(weights, scaled)
    calibration: Dict[str, object] = {"temperature": 1.0, "multipliers": [1.0] * len(EMOTIONS), "smoothing": 0.0}
    calibrated_score = float(blend_score)
    if not args.no_calibrate and y_true is not None:
        calibration = best_threshold_and_bias(blend_oof, y_true)
        calibrated_score = float(calibration["macro_f1"])
    calib_oof = apply_calibration(blend_oof, calibration) if y_true is not None else blend_oof

    # ----- test probabilities -------------------------------------------- #
    if test_stack is not None and test_order is not None:
        w = weights[keep_test]
        w = w / w.sum() if w.sum() > 0 else np.ones(len(keep_test)) / len(keep_test)
        blend_test = blend_with(w, test_stack)
        if not args.no_calibrate:
            blend_test = apply_calibration(blend_test, calibration)
        test_df = pd.DataFrame(blend_test, columns=_PROB_COLS)
        test_df.insert(0, "filename", test_order)
        test_df.to_csv(os.path.join(args.out, "test_probs.csv"), index=False, float_format="%.6f")
        np.save(os.path.join(args.out, "test_probs.npy"), blend_test.astype(np.float32))
        LOG.info("Wrote %s (%d rows)", os.path.join(args.out, "test_probs.csv"), len(test_df))
    else:
        LOG.warning("No test probability matrices available; only OOF artefacts were written")

    # ----- OOF artefacts + report ---------------------------------------- #
    oof_df = pd.DataFrame(blend_oof, columns=_PROB_COLS)
    oof_df.insert(0, "filename", order)
    if folds is not None and len(folds) == len(order):
        oof_df.insert(1, "fold", folds)
    if y_true is not None:
        oof_df.insert(1, "true", y_true)
    oof_df.to_csv(os.path.join(args.out, "oof_blend.csv"), index=False, float_format="%.6f")

    per_model = {
        tag: (float(macro_f1(y_true, stack[i].argmax(axis=1))) if y_true is not None else None)
        for i, tag in enumerate(tags)
    }
    payload = {
        "tags": tags,
        "weights": [float(w) for w in weights],
        "temperatures": [float(t) for t in temperatures],
        "calibration": calibration,
        "oof_macro_f1_raw": float(blend_score),
        "oof_macro_f1_calibrated": float(calibrated_score),
        "per_model_oof_macro_f1": per_model,
        "oof_files": oof_paths,
        "test_files": [p for p in test_paths if p],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": list(EMOTIONS),
        "n_oof": int(len(order)),
    }
    with open(os.path.join(args.out, "weights.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        f"DataVerse SER ensemble report ({payload['created']})",
        f"tags: {tags}",
        "",
        "per-model OOF macro-F1:",
        *[
            (f"  {tag:<20s} {per_model[tag]:.4f}" if per_model[tag] is not None else f"  {tag:<20s} n/a")
            for tag in tags
        ],
        "",
        f"weights:            {[round(float(w), 4) for w in weights]}",
        f"temperatures:       {[round(float(t), 3) for t in temperatures]}",
        f"blend OOF macro-F1: {float(blend_score):.4f}",
        f"calibrated OOF:     {float(calibrated_score):.4f}   (T={calibration.get('temperature')}, "
        f"multipliers={[round(float(m), 3) for m in calibration.get('multipliers', [])]}, "
        f"smoothing={calibration.get('smoothing')})",
    ]
    if y_true is not None:
        lines += ["", "per-class F1 (calibrated OOF):"]
        lines += [f"  {name:<8s} {score:.4f}" for name, score in per_class_f1(y_true, calib_oof.argmax(axis=1)).items()]
        if folds is not None and len(folds) == len(order):
            lines += ["", "per-fold macro-F1 (calibrated OOF):"]
            for fold in sorted(set(int(f) for f in folds)):
                mask = folds == fold
                lines.append(f"  fold {fold}: {macro_f1(y_true[mask], calib_oof[mask].argmax(axis=1)):.4f}")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "score_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
