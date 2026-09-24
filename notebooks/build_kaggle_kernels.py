#!/usr/bin/env python3
"""Generate self-contained Kaggle training kernels for DataVerse SER.

Each kernel embeds the current ``src/`` package + a per-model config, then
runs training + prediction on Kaggle's GPU (16 GB) with internet enabled so
weights download from HF Hub. Kernel outputs (OOF/test probs + checkpoints)
land in ``/kaggle/working`` and are pulled back locally via
``kaggle kernels output``.

Usage::

    python notebooks/build_kaggle_kernels.py            # regenerate kernels/
    DATAVERSE_DRYRUN=1 KAGGLE_WORKING=/tmp/opencode/ktest \\
        python kernels/xlsr/main.py                     # local smoke test
    kaggle kernels push -p kernels/xlsr                 # run on Kaggle
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent

COMPETITION = "dataverse-detecting-emotions-from-hindi-speech"
INPUT_ROOT = f"/kaggle/input/{COMPETITION}/SRCASW-BhavVaani"
WORKING = "/kaggle/working"
OWNER = "dezzki"

# model -> (config source, kaggle overrides)
MODELS: Dict[str, Dict] = {
    "xlsr": {
        "config": "configs/xlsr.yaml",
        "overrides": {
            "batch_size": 16, "grad_accum": 2, "grad_ckpt": False,
            "freeze_encoder_layers": 12, "dropout": 0.4,
            "label_smoothing": 0.1, "encoder_lr_scale": 0.05,
        },
    },
    "wavlm": {
        "config": "configs/wavlm.yaml",
        "overrides": {
            "batch_size": 16, "grad_accum": 2, "grad_ckpt": False,
            "freeze_encoder_layers": 12, "dropout": 0.4,
            "label_smoothing": 0.1, "encoder_lr_scale": 0.05,
        },
    },
    "hubert": {
        "config": "configs/hubert.yaml",
        "overrides": {
            "batch_size": 16, "grad_accum": 2, "grad_ckpt": False,
            "freeze_encoder_layers": 12, "dropout": 0.4,
            "label_smoothing": 0.1, "encoder_lr_scale": 0.05,
        },
    },
    # whisper_small retired from kernels: its weights aren't in the offline package.
    # ---- external-data variants (competition + 1,600 Hindi clips) ----
    "xlsr_aux": {
        "config": "configs/xlsr.yaml",
        "aux": True, "dataset_sources": ["vishlb/speech-emotion-recognition-hindi"],
        "overrides": {
            "batch_size": 16, "grad_accum": 4, "grad_ckpt": False,
            "freeze_encoder_layers": 12, "dropout": 0.4,
            "label_smoothing": 0.1, "encoder_lr_scale": 0.05,
        },
    },
    "wavlm_aux": {
        "config": "configs/wavlm.yaml",
        "aux": True, "dataset_sources": ["vishlb/speech-emotion-recognition-hindi"],
        "overrides": {
            "batch_size": 16, "grad_accum": 4, "grad_ckpt": False,
            "freeze_encoder_layers": 12, "dropout": 0.4,
            "label_smoothing": 0.1, "encoder_lr_scale": 0.05,
        },
    },
}

BOOTSTRAP = ""  # (removed)

KERNEL_MAIN = r'''"""DataVerse SER trainer (Kaggle). Generated from build_kaggle_kernels.py."""
import glob
import os
import pathlib
import subprocess
import sys

K = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
os.makedirs(K, exist_ok=True)
sys.path.insert(0, K)

PAYLOAD = __PAYLOAD__

# materialise src/ + config
src_dir = pathlib.Path(K) / "src"
src_dir.mkdir(exist_ok=True)
for name, content in PAYLOAD["src"].items():
    (src_dir / name).write_text(content, encoding="utf-8")
cfg_path = pathlib.Path(K) / PAYLOAD["config_name"]

# Locate the competition train.csv ANYWHERE under the input mount (layout varies).
INP_BASE = os.environ.get("DATAVERSE_INPUT_BASE", "/kaggle/input")
hits = sorted(glob.glob(os.path.join(INP_BASE, "**", "train.csv"), recursive=True))
if not hits:
    print("INPUT LAYOUT (csv files):",
          [p for p in glob.glob(INP_BASE + "/**/*.csv", recursive=True)][:20])
    raise SystemExit(f"train.csv not found under {INP_BASE}")
root = os.path.dirname(hits[0])
cfg_text = PAYLOAD["config_yaml"].replace("data_root: __ROOT__", f"data_root: {root}")
cfg_path.write_text(cfg_text, encoding="utf-8")

# Aux (external) data: turn relative manifest paths into absolute ones.
if PAYLOAD.get("aux") and PAYLOAD.get("aux_rows"):
    auxbase = os.environ.get("DATAVERSE_AUX_BASE", "/kaggle/input")
    first = PAYLOAD["aux_rows"][0]
    probe = glob.glob(os.path.join(auxbase, "**", first["path"]), recursive=True)
    if not probe:
        print("AUX INPUT LAYOUT:", [p for p in glob.glob(auxbase + "/**/*.wav", recursive=True)][:12])
        raise SystemExit("aux manifest path not found under " + auxbase)
    aux_root = probe[0][: -len(first["path"])]
    lines = ["filename,emotion,path"]
    for r in PAYLOAD["aux_rows"]:
        lines.append(",".join([r["filename"], r["emotion"], aux_root + r["path"]]))
    with open(os.path.join(K, "aux_hindi.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("AUX root =", aux_root, "| rows =", len(PAYLOAD["aux_rows"]))

# Model weights for OFFLINE training: extract the mounted models-dataset tarball.
import tarfile
MODELS_ROOT = f"{K}/models"
os.makedirs(MODELS_ROOT, exist_ok=True)
_found_tar = False
for _d in sorted(glob.glob(os.path.join(INP_BASE, "*"))):
    _tars = sorted(glob.glob(os.path.join(_d, "*.tar")))
    if _tars:
        for _t in _tars:
            print("extracting weights tar:", _t)
            with tarfile.open(_t, "r") as _tf:
                _tf.extractall(MODELS_ROOT)
        _found_tar = True
        break
if not _found_tar:
    print("WARN: no weights tarball found under", INP_BASE)

env = dict(
    os.environ,
    PYTHONPATH=K,
    HF_HOME=MODELS_ROOT,
    HF_HUB_CACHE=MODELS_ROOT,
    TRANSFORMERS_CACHE=MODELS_ROOT,
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
)

def cmd(*argv: str) -> None:
    subprocess.run(argv, env=env, check=True)

tag = PAYLOAD["tag"]

if os.environ.get("DATAVERSE_DRYRUN") == "1":
    cmd(sys.executable, "-m", "src.train_ssl", "--config", str(cfg_path), "--dry-run",
        "--device", os.environ.get("DATAVERSE_DEVICE", "auto"))
    print("DATAVERSE_DRYRUN OK")
    sys.exit(0)

# 3. train every fold
cmd(sys.executable, "-m", "src.train_ssl", "--config", str(cfg_path),
    "--out", f"{K}/runs/{tag}", "--folds", "0", "1", "2", "3", "4")

# 4. predict: OOF + test probabilities
cmd(sys.executable, "-m", "src.predict", "--config", str(cfg_path),
    "--ckpt", f"{K}/runs/{tag}", "--folds", "0", "1", "2", "3", "4")

# 5. report
print("=== KAGGLE RUN COMPLETE ===")
for rootpath, _, files in os.walk(K):
    for f in files:
        p = os.path.join(rootpath, f)
        if os.path.getsize(p) > 0:
            print(p)
'''


def build_payload(model: str, spec: Dict) -> Dict:
    import yaml

    cfg = yaml.safe_load((ROOT / spec["config"]).read_text())
    cfg["data_root"] = "__ROOT__"  # resolved at runtime on Kaggle
    cfg["output_dir"] = f"{WORKING}/runs"
    cfg["oof_dir"] = f"{WORKING}/oof"
    cfg["num_workers"] = 2
    cfg["folds"] = [0, 1, 2, 3, 4]
    cfg.update(spec["overrides"])
    tag = str(cfg["model_name"]).rstrip("/").split("/")[-1]
    aux = bool(spec.get("aux"))
    aux_rows: List[Dict] = []
    if aux:
        tag = f"{tag}_aux"
        cfg["aux_root"] = f"{WORKING}"
        cfg["aux_csv"] = "aux_hindi.csv"
        man = ROOT / "ext" / "aux_hindi.csv"
        if man.exists():
            import csv as _csv
            with open(man, newline="", encoding="utf-8") as fh:
                aux_rows = [dict(r) for r in _csv.DictReader(fh)]
    src: Dict[str, str] = {}
    for p in sorted((ROOT / "src").glob("*.py")):
        src[p.name] = p.read_text()
    return {
        "src": src,
        "config_name": f"{model}.yaml",
        "config_yaml": yaml.safe_dump(cfg, sort_keys=False),
        "tag": tag,
        "aux": aux,
        "aux_rows": aux_rows,
    }


def write_payload_literal(payload: Dict) -> str:
    return repr(payload)  # Python literal (True/False/None), safe to inline


def generate() -> None:
    out_root = ROOT / "kernels"
    out_root.mkdir(exist_ok=True)
    for model, spec in MODELS.items():
        payload = build_payload(model, spec)
        main_py = KERNEL_MAIN.replace("__PAYLOAD__", write_payload_literal(payload))
        kernel_dir = out_root / model
        kernel_dir.mkdir(exist_ok=True)
        (kernel_dir / "main.py").write_text(main_py, encoding="utf-8")
        meta = {
            "id": f"{OWNER}/dataverse-{model}",
            "title": f"dataverse-{model}",
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "enable_tpu": False,
            "competition_sources": [COMPETITION],
        }
        ds = ["dezzki/dataverse-hf-models"]
        if spec.get("dataset_sources"):
            ds += [s for s in spec["dataset_sources"] if s not in ds]
        meta["dataset_sources"] = ds
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"generated kernels/{model}/ (tag={payload['tag']})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    generate()