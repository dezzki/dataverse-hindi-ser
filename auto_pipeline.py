#!/usr/bin/env python3
"""DataVerse auto-pipeline watchdog.

Runs unattended: waits for local + Kaggle model runs, pulls outputs, builds
the calibrated ensemble, and submits (respecting the 5/day Kaggle cap).
Logs every step to context.md. Safe to run as a loop or --once.

Safeguards:
  * never submits when today's Kaggle quota is 0
  * never re-submits the same source-signature
  * no new GPU work is launched (only consumes finished runs)
State is kept in runs/final/.state.json so the loop is idempotent.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional

ROOT = "/home/dezzki/Projects/DataVerse"
COMP = "dataverse-detecting-emotions-from-hindi-speech"
STATE = f"{ROOT}/runs/final/.state.json"
FIN = f"{ROOT}/runs/final"
CTX = f"{ROOT}/context.md"
PY = f"{ROOT}/.venv/bin/python"

# slug -> tag
KERNELS: Dict[str, str] = {
    "dezzki/dataverse-xlsr": "wav2vec2-large-xlsr-53",
    "dezzki/dataverse-wavlm": "wavlm-large",
    "dezzki/dataverse-hubert": "hubert-large-ls960-ft",
    "dezzki/dataverse-whisper-small": "whisper-small",
}
LOCAL_TAG = "wav2vec2-base-superb-er"
LOCAL_RUN = f"{ROOT}/runs/{LOCAL_TAG}"
PREDICT_CONFIG = "configs/base_superb.yaml"
SEEDS = ["classic", LOCAL_TAG] + list(KERNELS.values())


def log(msg: str) -> None:
    line = f"| 2026-09-25 | auto | {msg} |"
    with open(CTX, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(msg, flush=True)


def sh(cmd: str, timeout: int = 600) -> str:
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return f"{r.stdout}\n{r.stderr}".strip()


def read_state() -> Dict:
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"submitted": "", "last_oof": None}


def write_state(st: Dict) -> None:
    os.makedirs(FIN, exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=2)


def kernel_status(slug: str) -> str:
    out = sh(f"kaggle kernels status {slug}")
    for tok in out.split('"'):
        if "status" not in tok.lower() and ("RUNNING" in tok or "COMPLETE" in tok or "ERROR" in tok or "QUEUED" in tok or "PENDING" in tok or "CANCELLED" in tok):
            return tok.replace("KernelWorkerStatus.", "").lower()
    return out[:80].strip()


def pull_kernel(slug: str, tag: str) -> bool:
    dl = f"{ROOT}/kaggle_dl/{tag}"
    os.makedirs(dl, exist_ok=True)
    sh(f"kaggle kernels output {slug} -p {dl} -q", timeout=900)
    dst = f"{ROOT}/oof/{tag}"
    os.makedirs(dst, exist_ok=True)
    moved = False
    for f in glob.glob(f"{dl}/**/*.csv", recursive=True):
        name = os.path.basename(f)
        if name.startswith("oof_probs_") or name.startswith("test_probs_"):
            shutil.copy2(f, os.path.join(dst, name))
            moved = True
    return moved


def local_superb_ready() -> bool:
    return all(os.path.exists(f"{LOCAL_RUN}/fold{i}/best.pt") for i in range(5))


def have_oof(tag: str) -> bool:
    return bool(glob.glob(f"{ROOT}/oof/**/oof_probs_{tag}.csv", recursive=True))


def ensure_classic_oof() -> None:
    """Normalise classical OOF/test probs into oof/classic/ for tag-based mixing."""
    src = f"{ROOT}/runs/classic"
    dst = f"{ROOT}/oof/classic"
    os.makedirs(dst, exist_ok=True)
    for name in ("oof_probs_classic.csv", "test_probs_classic.csv"):
        if not os.path.exists(f"{dst}/{name}") and os.path.exists(f"{src}/{name}"):
            shutil.copy2(f"{src}/{name}", f"{dst}/{name}")
            log(f"normalised {name} -> oof/classic/")


def remaining_submissions() -> int:
    out = sh(f"kaggle competitions submission-limits -c {COMP}")
    m = [l for l in out.splitlines() if "remaining" in l.lower() or "Remaining" in l]
    for l in m:
        for tok in l.split():
            if tok.isdigit():
                return int(tok)
    return 0


def re_first(text: str, pat: str) -> Optional[str]:
    import re
    m = re.search(pat, text)
    return m.group(1) if m else None


def build_ensemble(tags: List[str]) -> Optional[float]:
    os.makedirs(FIN, exist_ok=True)
    r = sh(f"{PY} -m src.ensemble --tags {' '.join(tags)} --out {FIN}", timeout=1800)
    m = re_first(r, r"calibrated OOF:\s+([0-9.]+)")
    log(f"ensemble over [{', '.join(tags)}] -> calibrated OOF {m if m else 'n/a'}")
    return float(m) if m else None


def submit_current(tag_list: List[str]) -> Optional[str]:
    if remaining_submissions() <= 0:
        log("SKIP submit: no daily submissions remaining")
        return None
    # build + validate submission
    r = sh(f"{PY} -m src.submit --probs {FIN}/test_probs.csv --data-root data/SRCASW-BhavVaani --out {FIN}/submission.csv", timeout=300)
    if "RESULT: PASS" not in r:
        log(f"submit.py FAILED: {r[-400:]}")
        return None
    label = f"auto ensemble [{','.join(tag_list)}]"
    out = sh(f"kaggle competitions submit -c {COMP} -f {FIN}/submission.csv -m '{label}'", timeout=300)
    log(f"SUBMITTED ({label}): {out[-200:].replace(chr(10),' ')}")
    return label


def once() -> None:
    st = read_state()
    ensure_classic_oof()

    # 1) local superb -> predict when folds all done
    if local_superb_ready() and not have_oof(LOCAL_TAG):
        log("local superb folds complete -> running predict")
        sh(f"{PY} -m src.predict --config {PREDICT_CONFIG} --ckpt {LOCAL_RUN} --folds 0 1 2 3 4", timeout=5400)

    # 2) pull finished Kaggle kernels
    for slug, tag in KERNELS.items():
        status = kernel_status(slug)
        if status in ("running", "queued", "pending"):
            log(f"kernel {slug}: {status} (not pulled)")
        elif status in ("complete", "completed"):
            if not have_oof(tag):
                moved = pull_kernel(slug, tag)
                log(f"kernel {slug}: COMPLETE, pulled outputs (moved={moved})")
            else:
                log(f"kernel {slug}: complete (already pulled)")
        elif status in ("error", "failed", "cancelled"):
            log(f"kernel {slug}: **{status.upper()}** — needs action")
        else:
            log(f"kernel {slug}: status={status!r}")

    # 3) ensemble + submit when a new source is available
    available = [t for t in SEEDS if have_oof(t)]
    log(f"available OOF sources: {available}")
    if available:
        sig = ";".join(sorted(available))
        m = build_ensemble(available)
        if sig != st.get("submitted", "") and m is not None:
            label = submit_current(available)
            if label:
                st["submitted"] = sig
                st["last_oof"] = m
                write_state(st)
                sh("git add -A && git commit -q -m 'auto: ensemble submit' || true", timeout=120)
    # else: keep waiting (no complete sources yet)


def loop(interval: int = 300, once_only: bool = False) -> None:
    if once_only:
        return once()
    while True:
        try:
            once()
        except Exception as exc:
            log(f"auto iteration error: {exc!r}")
        time.sleep(interval)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIN, exist_ok=True)
    loop(a.interval, a.once)