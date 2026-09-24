#!/usr/bin/env python3
"""Build an auxiliary (external) SER manifest from a labelled audio tree.

Walks ``DATASET_ROOT/<actor>/session*/<emotion>/*.wav`` and emits a CSV with
columns ``filename, emotion, relpath`` where ``emotion`` is remapped onto the
competition label space via ``LABEL_MAP`` (non-mapped emotions are dropped).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "anger": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    # disgust / fear / sarcastic / surprise -> not in the 4-class space -> dropped
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root (e.g. 'ext/my Dataset').")
    ap.add_argument("--out", default="ext/aux_hindi.csv")
    a = ap.parse_args()

    root = Path(a.root)
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root).as_posix()
        parts = rel.split("/")
        if len(parts) < 3:
            continue
        emotion_dir = parts[-2].lower()
        if emotion_dir not in LABEL_MAP:
            continue
        rows.append({"filename": wav.name, "emotion": LABEL_MAP[emotion_dir], "path": rel})

    df = pd.DataFrame(rows)
    print("total wavs (4-class):", len(df))
    print(df["emotion"].value_counts().to_dict())
    if len(df):
        df.to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()