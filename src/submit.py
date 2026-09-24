"""Build and validate ``submission.csv`` for the Kaggle Hindi SER competition.

Hard requirements enforced here (a mismatch means disqualification):

* exact header ``Id,emotion``
* one row per ``test.csv`` ``Id``, same count, same set (no missing/extra ids)
* labels lowercase and inside ``{angry, happy, neutral, sad}``
* rows sorted by ``Id``

Usage
-----
::

    python -m src.submit --probs runs/final/test_probs.csv
    python -m src.submit --probs oof/superb/test_probs_superb.csv --out submission.csv

Exit code is non-zero when validation FAILs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import EMOTIONS, setup_logging

LOG = logging.getLogger("submit")

_PROB_COLS = [f"prob_{e}" for e in EMOTIONS]

_ALLOWED_EXTRA = {"filename", "Id", "emotion", "fold", "true", "pred", "confidence"} | set(_PROB_COLS)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_test_csv(data_root: str, test_csv: str) -> str:
    path = test_csv if os.path.isabs(test_csv) else os.path.join(data_root, test_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"test.csv not found at {path}. Pass --test-csv or --data-root (Kaggle: "
            "'/kaggle/input/<competition>')."
        )
    return path


def predictions_from_probs(probs_path: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read a predictions file -> ``(frame, key, warnings)``.

    ``key`` is ``"filename"`` for probability matrices (``prob_anger..prob_sad``)
    and for ``emotion`` CSVs that carry a ``filename`` column, or ``"Id"`` when
    the file is already in submission space (``Id,emotion``).
    """
    warnings: List[str] = []
    df = pd.read_csv(probs_path)
    if "emotion" in df.columns:
        if "filename" in df.columns:
            key = "filename"
            values = df["filename"].astype(str)
        elif "Id" in df.columns:
            key = "Id"
            values = df["Id"]
            warnings.append(f"{probs_path} is already keyed by Id; validating it directly against test.csv")
        else:
            raise ValueError(f"{probs_path} needs a 'filename' or 'Id' column alongside 'emotion'")
        out = pd.DataFrame({key: values, "emotion": df["emotion"].astype(str).str.strip().str.lower()})
        if len(out) != len(out.drop_duplicates(key)):
            raise ValueError(f"{probs_path} has duplicated {key} values")
        return out, key, warnings

    missing = [c for c in _PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{probs_path} needs either an 'emotion' column or the probability columns {missing}"
        )
    if "filename" not in df.columns:
        raise ValueError(f"{probs_path} must contain a 'filename' column")
    probs = df[_PROB_COLS].to_numpy(dtype=np.float64)
    labels = [EMOTIONS[i] for i in probs.argmax(axis=1)]
    out = pd.DataFrame({"filename": df["filename"].astype(str), "emotion": labels})
    if df["filename"].duplicated().any():
        raise ValueError(f"{probs_path} has duplicated filenames")
    return out, "filename", warnings


def build_submission(
    preds: pd.DataFrame,
    test_csv_path: str,
    probs_path: str,
    key: str = "filename",
) -> Tuple[pd.DataFrame, List[str]]:
    """Join predictions with ``test.csv``; return (submission, problems)."""
    test = pd.read_csv(test_csv_path)
    if "Id" not in test.columns:
        raise ValueError(f"{test_csv_path} must contain an 'Id' column")
    if key == "filename" and "filename" not in test.columns:
        raise ValueError(f"{test_csv_path} must contain a 'filename' column")

    problems: List[str] = []

    if key == "Id":
        left = pd.DataFrame({"Id": test["Id"], "_key": test["Id"].astype(str)})
        right = pd.DataFrame(
            {"_key": preds["Id"].astype(str), "emotion": preds["emotion"].astype("object")}
        ).drop_duplicates("_key")
        merged = left.merge(right, on="_key", how="left")
        merged = merged.rename(columns={"emotion": "emotion"})
        missing = merged["emotion"].isna()
        if missing.any():
            problems.append(
                f"{int(missing.sum())} test Id(s) have no prediction (e.g. "
                f"{merged.loc[missing, 'Id'].head(3).tolist()}) from {probs_path}"
            )
        extra_ids = sorted(set(right["_key"]) - set(left["_key"]))
        if extra_ids:
            problems.append(f"{len(extra_ids)} predicted Id(s) are not in test.csv (e.g. {extra_ids[:3]})")
        submission = pd.DataFrame({"Id": merged["Id"], "emotion": merged["emotion"]})
    else:
        test = pd.DataFrame(
            {
                "Id": test["Id"],
                "filename": test["filename"].astype(str).map(
                    lambda s: os.path.basename(str(s).replace("\\", "/"))
                ),
            }
        )
        preds = preds.copy()
        preds["filename"] = preds["filename"].map(lambda s: os.path.basename(str(s).replace("\\", "/")))
        merged = test.merge(preds, on="filename", how="left")
        missing = merged["emotion"].isna()
        if missing.any():
            problems.append(
                f"{int(missing.sum())} test file(s) have no prediction (e.g. "
                f"{merged.loc[missing, 'filename'].head(3).tolist()}) from {probs_path}"
            )
        extras = sorted(set(preds["filename"]) - set(test["filename"]))
        if extras:
            problems.append(f"{len(extras)} predicted file(s) are not in test.csv (e.g. {extras[:3]})")
        submission = pd.DataFrame({"Id": merged["Id"], "emotion": merged["emotion"]})

    # Lowercase real predictions but keep NaNs (they must surface as FAIL).
    emotion = submission["emotion"].astype("object")
    emotion = emotion.where(emotion.isna(), emotion.astype(str).str.strip().str.lower())
    submission = pd.DataFrame({"Id": submission["Id"], "emotion": emotion})
    try:  # numeric Ids sort numerically, otherwise lexicographically
        submission = submission.assign(_sort=submission["Id"].astype("int64")).sort_values("_sort")
        submission = submission.drop(columns="_sort")
    except (TypeError, ValueError):
        submission = submission.sort_values("Id", kind="stable")
    return submission.reset_index(drop=True), problems
    try:  # numeric Ids sort numerically, otherwise lexicographically
        submission = submission.assign(_sort=submission["Id"].astype("int64")).sort_values("_sort")
        submission = submission.drop(columns="_sort")
    except (TypeError, ValueError):
        submission = submission.sort_values("Id", kind="stable")
    submission = submission.reset_index(drop=True)
    return submission, [], problems


def validate_submission(submission: pd.DataFrame, test_csv_path: str) -> Tuple[bool, List[str]]:
    """Return ``(ok, checks)`` where checks is a list of human-readable lines."""
    test = pd.read_csv(test_csv_path)
    checks: List[str] = []
    ok = True

    def check(condition: bool, message: str) -> None:
        nonlocal ok
        checks.append(("PASS  " if condition else "FAIL  ") + message)
        ok = ok and bool(condition)

    check(list(submission.columns) == ["Id", "emotion"], f"header is exactly ['Id', 'emotion'] (got {list(submission.columns)})")
    check(len(submission) == len(test), f"row count {len(submission)} == len(test.csv) {len(test)}")
    check(not submission["Id"].duplicated().any(), "no duplicated Ids")

    test_ids = set(test["Id"].tolist())
    pred_ids = set(submission["Id"].tolist())
    check(pred_ids == test_ids, f"Id set matches test.csv exactly (missing={len(test_ids - pred_ids)}, extra={len(pred_ids - test_ids)})")

    check(not submission["emotion"].isna().any(), "every row has a label (no missing predictions)")
    labels = set(submission["emotion"].dropna().astype(str).str.strip().str.lower().unique().tolist())
    check(labels.issubset(set(EMOTIONS)), f"labels subset of {list(EMOTIONS)} (got {sorted(labels)})")
    check(
        submission["emotion"].dropna().astype(str).tolist()
        == submission["emotion"].dropna().astype(str).str.lower().tolist(),
        "all labels lowercase",
    )
    return ok, checks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.submit",
        description="Build submission.csv (Id,emotion) from a test-probabilities CSV and validate it.",
    )
    parser.add_argument("--probs", required=True, help="test_probs CSV (filename + prob_*) or an Id,emotion CSV.")
    parser.add_argument("--out", default="submission.csv", help="Output path (default: submission.csv).")
    parser.add_argument("--data-root", default="data", help="Directory holding test.csv (Kaggle: /kaggle/input/<comp>).")
    parser.add_argument("--test-csv", default="test.csv", help="test.csv name or absolute path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    test_csv_path = _resolve_test_csv(args.data_root, args.test_csv)
    preds, key, warns = predictions_from_probs(args.probs)
    for warning in warns:
        LOG.info(warning)

    submission, problems = build_submission(preds, test_csv_path, args.probs, key=key)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    submission.to_csv(args.out, index=False)

    ok, checks = validate_submission(submission, test_csv_path)
    print("submission validation:")
    for line in checks:
        print("  " + line)
    for problem in problems:
        print("  WARN  " + problem)
    print(f"wrote {args.out} ({len(submission)} rows)")

    if not ok:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
