"""Classical SER baseline: eGeMAPS (+ MFCC stats) -> StandardScaler -> SVM / LightGBM.

Runs stratified 5-fold CV with the shared fold manifest, reports macro-F1 and
writes OOF/test probability CSVs in exactly the same convention as the SSL
pipeline so the classical models can be blended by :mod:`src.ensemble`.

Usage
-----
::

    python -m src.classic --config configs/classic.yaml
    python -m src.classic --config configs/classic.yaml --models sklearn_svm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .config import EMOTIONS, EMOTION_TO_IDX, ExperimentConfig, load_config, setup_logging
from .data import read_test_df, read_train_df, split_summary
from .eval import classification_report, macro_f1, per_class_f1, plot_confusion
from .features import build_classic_features
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


LOG = logging.getLogger("classic")

#: default model registry (kept small and dependency-light)
DEFAULT_MODELS = ["sklearn_svm", "lightgbm"]
_PROB_COLS = [f"prob_{e}" for e in EMOTIONS]


# --------------------------------------------------------------------------- #
# Model builders (heavy imports are lazy so this module always imports)
# --------------------------------------------------------------------------- #
def build_estimator(
    name: str,
    cfg: ExperimentConfig,
    seed: int,
    n_features: int,
    n_samples: Optional[int] = None,
):
    """Return a fresh sklearn-compatible estimator (probability-enabled).

    ``n_samples`` is used to relax LightGBM's ``min_child_samples`` on tiny
    datasets (the full competition data is large enough to keep the default).
    """
    options: Dict[str, Any] = dict(cfg.classic or {})
    if name in ("sklearn_svm", "svm", "svc"):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=float(options.get("svm_c", 2.0)),
                        gamma=options.get("svm_gamma", "scale"),
                        kernel=options.get("svm_kernel", "rbf"),
                        class_weight=options.get("class_weight", "balanced"),
                        probability=True,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    if name in ("lightgbm", "lgbm"):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "lightgbm is required for the LightGBM baseline. Install with: pip install lightgbm"
            ) from exc

        min_child_samples = int(options.get("lgbm_min_child_samples", 20))
        if n_samples:
            min_child_samples = max(2, min(min_child_samples, max(2, int(n_samples) // 20)))
        return LGBMClassifier(
            n_estimators=int(options.get("lgbm_n_estimators", 400)),
            learning_rate=float(options.get("lgbm_learning_rate", 0.05)),
            num_leaves=int(options.get("lgbm_num_leaves", 31)),
            min_child_samples=min_child_samples,
            subsample=float(options.get("lgbm_subsample", 0.9)),
            colsample_bytree=float(options.get("lgbm_colsample", 0.8)),
            class_weight=options.get("class_weight", "balanced"),
            random_state=int(seed),
            n_jobs=int(options.get("lgbm_n_jobs", -1)),
            verbose=-1,
        )
    if name in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for the XGBoost baseline. Install with: pip install xgboost") from exc

        return XGBClassifier(
            n_estimators=int(options.get("xgb_n_estimators", 400)),
            learning_rate=float(options.get("xgb_learning_rate", 0.05)),
            max_depth=int(options.get("xgb_max_depth", 6)),
            subsample=float(options.get("xgb_subsample", 0.9)),
            colsample_bytree=float(options.get("xgb_colsample", 0.8)),
            random_state=int(seed),
            n_jobs=int(options.get("xgb_n_jobs", -1)),
            eval_metric="mlogloss",
            tree_method="hist",
        )
    raise ValueError(f"Unknown classical model {name!r}; expected one of {DEFAULT_MODELS + ['xgboost']}")


def _predict_proba(estimator, X: np.ndarray, n_classes: int) -> np.ndarray:
    """``predict_proba`` padded/ordered to the full 4-class label space."""
    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        classes = np.asarray(getattr(estimator, "classes_", np.arange(probs.shape[1])), dtype=int)
    else:  # pragma: no cover - defensive
        from sklearn.preprocessing import label_binarize

        decision = np.asarray(estimator.decision_function(X), dtype=np.float64)
        probs = label_binarize(np.argmax(decision, axis=1), classes=list(range(n_classes))).astype(np.float64)
        classes = np.arange(n_classes)
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
    for j, cls in enumerate(classes):
        if 0 <= int(cls) < n_classes:
            out[:, int(cls)] = probs[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


# --------------------------------------------------------------------------- #
# CV loop
# --------------------------------------------------------------------------- #
def run_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: Optional[np.ndarray],
    cfg: ExperimentConfig,
    fold_ids: Sequence[int],
) -> Dict[str, Any]:
    """Stratified CV for one estimator; returns OOF/test probs + fold scores."""
    n_classes = len(EMOTIONS)
    oof = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_scores: Dict[str, float] = {}
    test_probs: List[np.ndarray] = []

    for fold in fold_ids:
        train_mask = folds != int(fold)
        val_mask = folds == int(fold)
        estimator = build_estimator(
            name, cfg, seed=int(cfg.seed) + int(fold), n_features=X.shape[1], n_samples=int(train_mask.sum())
        )
        estimator.fit(X[train_mask], y[train_mask])
        oof[val_mask] = _predict_proba(estimator, X[val_mask], n_classes)
        fold_scores[fold] = macro_f1(y[val_mask], oof[val_mask].argmax(axis=1))
        LOG.info("%s | fold %d | val macro-F1 %.4f (n=%d)", name, fold, fold_scores[fold], int(val_mask.sum()))
        if X_test is not None:
            test_probs.append(_predict_proba(estimator, X_test, n_classes))

    if X_test is not None:
        full = build_estimator(name, cfg, seed=int(cfg.seed), n_features=X.shape[1], n_samples=len(y))
        full.fit(X, y)
        test_probs.append(_predict_proba(full, X_test, n_classes))

    return {
        "name": name,
        "oof": oof.astype(np.float32),
        "test": np.mean(np.stack(test_probs, axis=0), axis=0).astype(np.float32) if test_probs else None,
        "fold_scores": fold_scores,
        "mean_f1": float(np.mean(list(fold_scores.values()))) if fold_scores else float("nan"),
        "std_f1": float(np.std(list(fold_scores.values()))) if fold_scores else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_oof_csv(path: str, filenames: Sequence[str], folds: Sequence[int], y: Sequence[int], probs: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"filename": list(filenames), "fold": list(folds), "true": list(y)})
    prob_cols = pd.DataFrame(probs, columns=_PROB_COLS)
    pd.concat([df, prob_cols], axis=1).to_csv(path, index=False, float_format="%.6f")


def write_test_csv(path: str, filenames: Sequence[str], probs: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(probs, columns=_PROB_COLS)
    df.insert(0, "filename", list(filenames))
    df.to_csv(path, index=False, float_format="%.6f")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.classic",
        description="Train classical SER baselines (eGeMAPS/MFCC -> SVM/LightGBM) with 5-fold CV.",
    )
    parser.add_argument("--config", default="configs/classic.yaml", help="YAML config.")
    parser.add_argument("--out", default=None, help="Output dir (default: cfg.output_dir/classic).")
    parser.add_argument("--models", nargs="*", default=None, help="Override cfg.classic_models.")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Folds to run (default: cfg.folds).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore/won't write cached feature matrices.")
    parser.add_argument("--plot", action="store_true", help="Also save confusion-matrix PNGs.")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed.")
    parser.add_argument("--folds-json", default=None, help="Fold manifest (default: <data_root>/folds.json).")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Config overrides, e.g. --set data_root=data/SRCASW-BhavVaani",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    overrides = {"seed": args.seed} if args.seed is not None else {}
    overrides.update(_parse_set_args(args.set, build_parser()))
    cfg = load_config(args.config, overrides)
    if args.no_cache:
        cfg.features = {**cfg.features, "cache": False}
    out_dir = args.out or os.path.join(cfg.output_dir, "classic")
    os.makedirs(out_dir, exist_ok=True)
    models = list(args.models) if args.models else list(cfg.classic_models or DEFAULT_MODELS)
    fold_ids = list(args.folds) if args.folds else list(cfg.folds)

    # ----- data + folds ---------------------------------------------------- #
    train_df = read_train_df(cfg.data_root, cfg)
    test_df = read_test_df(cfg.data_root, cfg)
    LOG.info(split_summary(train_df, test_df))
    folds_path = args.folds_json or fold_assignment_path(cfg.data_root, out_dir)
    train_df = load_or_create_folds(train_df, cfg.num_folds, cfg.seed, path=folds_path)

    LOG.info("Extracting classical features (this can take a few minutes) ...")
    train_feats = build_classic_features(train_df, cfg.data_root, cfg, split="train")
    test_feats = build_classic_features(test_df, cfg.data_root, cfg, split="test")

    feature_cols = [c for c in train_feats.columns if c != "filename"]
    # align by filename (robust against any row reordering)
    merged_train = train_df[["filename", "emotion", "fold"]].merge(train_feats, on="filename", how="left")
    merged_test = test_df[["filename"]].merge(test_feats, on="filename", how="left")
    if merged_train[feature_cols].isna().all(axis=1).any():
        LOG.warning("Some train rows have no features (all-NaN); they will be zero-filled")
    X = merged_train[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    X_test = merged_test[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = np.asarray([EMOTION_TO_IDX[e] for e in merged_train["emotion"].astype(str)], dtype=np.int64)
    folds = merged_train["fold"].to_numpy(dtype=np.int64)
    LOG.info("Feature matrices: train %s, test %s", X.shape, X_test.shape)

    # ----- models ---------------------------------------------------------- #
    results: List[Dict[str, Any]] = []
    enabled_blocks = [b for b in ("opensmile", "mfcc") if (cfg.features or {}).get(b)]
    report_lines = [
        "DataVerse SER - classical baseline report",
        f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"config:  {args.config}",
        f"data:    {cfg.data_root}  ({len(train_df)} train / {len(test_df)} test)",
        f"features: {len(feature_cols)} columns (blocks: {' + '.join(enabled_blocks) or 'none'})",
        "",
    ]
    for name in models:
        try:
            result = run_cv(name, X, y, folds, X_test, cfg, fold_ids)
        except RuntimeError as exc:  # missing optional dependency
            LOG.warning("Skipping %s: %s", name, exc)
            report_lines.append(f"{name}: SKIPPED ({exc})")
            continue
        results.append(result)

        report_lines += [
            f"=== {name} ===",
            "per-fold macro-F1:",
            *[f"  fold {f}: {s:.4f}" for f, s in sorted(result["fold_scores"].items())],
            f"mean macro-F1: {result['mean_f1']:.4f} (+/- {result['std_f1']:.4f})",
            classification_report(y, result["oof"].argmax(axis=1), title=f"{name} OOF"),
            "",
        ]

        write_oof_csv(os.path.join(out_dir, f"oof_probs_classic_{name}.csv"), merged_train["filename"], folds, y, result["oof"])
        write_test_csv(
            os.path.join(out_dir, f"test_probs_classic_{name}.csv"),
            merged_test["filename"],
            result["test"] if result["test"] is not None else np.full((len(merged_test), len(EMOTIONS)), 1 / len(EMOTIONS)),
        )
        if args.plot:
            plot_confusion(
                y, result["oof"].argmax(axis=1),
                os.path.join(out_dir, f"confusion_{name}.png"), title=f"{name} OOF",
            )

    if not results:
        LOG.error("No classical model could be trained.")
        return 2

    best = max(results, key=lambda r: r["mean_f1"])
    write_oof_csv(os.path.join(out_dir, "oof_probs.csv"), merged_train["filename"], folds, y, best["oof"])
    write_oof_csv(os.path.join(out_dir, "oof_probs_classic.csv"), merged_train["filename"], folds, y, best["oof"])
    if best["test"] is not None:
        write_test_csv(os.path.join(out_dir, "test_probs.csv"), merged_test["filename"], best["test"])
        write_test_csv(os.path.join(out_dir, "test_probs_classic.csv"), merged_test["filename"], best["test"])
    report_lines += [
        f"best model: {best['name']} (mean macro-F1 {best['mean_f1']:.4f})",
        "ensemble hand-off:",
        f"  --tags classic  (dir {out_dir}, tag 'classic' files oof_probs_classic.csv / test_probs_classic.csv)",
        f"  or per-model: oof_probs_classic_{best['name']}.csv / test_probs_classic_{best['name']}.csv",
        "",
    ]
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "models": {
                    r["name"]: {"mean_macro_f1": r["mean_f1"], "std_macro_f1": r["std_f1"], "folds": r["fold_scores"]}
                    for r in results
                },
                "best": best["name"],
                "n_features": len(feature_cols),
                "fold_ids": [int(f) for f in fold_ids],
            },
            fh,
            indent=2,
        )
    print("\n".join(report_lines))
    LOG.info("Wrote %s", os.path.join(out_dir, "report.txt"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
