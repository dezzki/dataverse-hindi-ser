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
        "overrides": {"batch_size": 16, "grad_accum": 2, "grad_ckpt": False},
    },
    "wavlm": {
        "config": "configs/wavlm.yaml",
        "overrides": {"batch_size": 16, "grad_accum": 2, "grad_ckpt": False},
    },
    "hubert": {
        "config": "configs/hubert.yaml",
        "overrides": {"batch_size": 16, "grad_accum": 2, "grad_ckpt": False},
    },
    "whisper_small": {
        "config": "configs/whisper_small.yaml",
        "overrides": {"batch_size": 20, "grad_accum": 2, "grad_ckpt": False},
    },
}

BOOTSTRAP = ""  # (removed)

KERNEL_MAIN = r'''"""DataVerse SER trainer (Kaggle). Generated from build_kaggle_kernels.py."""
import os
import pathlib
import subprocess
import sys

K = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
os.makedirs(K, exist_ok=True)
sys.path.insert(0, K)

PAYLOAD = __PAYLOAD__

# 1. light dep refresh (best effort)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers", "librosa", "soundfile", "safetensors", "accelerate"],
    check=False,
    env=dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1"),
)

# 2. materialise src/ + config
src_dir = pathlib.Path(K) / "src"
src_dir.mkdir(exist_ok=True)
for name, content in PAYLOAD["src"].items():
    (src_dir / name).write_text(content, encoding="utf-8")
cfg_path = pathlib.Path(K) / PAYLOAD["config_name"]
cfg_path.write_text(PAYLOAD["config_yaml"], encoding="utf-8")

env = dict(
    os.environ,
    PYTHONPATH=K,
    HF_HOME=f"{K}/hf",
    TRANSFORMERS_CACHE=f"{K}/hf",
    HUGGINGFACE_HUB_CACHE=f"{K}/hf",
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
for root, _, files in os.walk(K):
    for f in files:
        p = os.path.join(root, f)
        if os.path.getsize(p) > 0:
            print(p)
'''


def build_payload(model: str, spec: Dict) -> Dict:
    import yaml

    cfg = yaml.safe_load((ROOT / spec["config"]).read_text())
    cfg["data_root"] = INPUT_ROOT
    cfg["output_dir"] = f"{WORKING}/runs"
    cfg["oof_dir"] = f"{WORKING}/oof"
    cfg["num_workers"] = 2
    cfg["folds"] = [0, 1, 2, 3, 4]
    cfg.update(spec["overrides"])
    tag = str(cfg["model_name"]).rstrip("/").split("/")[-1]
    src: Dict[str, str] = {}
    for p in sorted((ROOT / "src").glob("*.py")):
        src[p.name] = p.read_text()
    return {
        "src": src,
        "config_name": f"{model}.yaml",
        "config_yaml": yaml.safe_dump(cfg, sort_keys=False),
        "tag": tag,
    }


def write_payload_literal(payload: Dict) -> str:
    return json.dumps(payload, indent=2, default=str)


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
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"generated kernels/{model}/ (tag={payload['tag']})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    generate()