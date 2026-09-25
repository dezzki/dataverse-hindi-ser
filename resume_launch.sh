#!/usr/bin/env bash
# Resume local training: WavLM folds 1-4 (fold0 done), then HuBERT all, then XLSR(+aux) all.
# Detached with setsid so it survives session-initiated process-group reaping.
set -u
cd "$(dirname "$0")"
VENV=.venv/bin/python
export PYTHONUNBUFFERED=1

# watchdog health (re-launch if missing) is handled externally; ensure dirs
mkdir -p runs oof

echo "$(date) resume: WavLM folds 1-4"
"$VENV" -m src.train_ssl --config configs/wavlm_local.yaml --folds 1 2 3 4
echo "$(date) resume: HuBERT all folds"
"$VENV" -m src.train_ssl --config configs/hubert_local.yaml --folds 0 1 2 3 4
echo "$(date) resume: XLSR(+aux) all folds"
"$VENV" -m src.train_ssl --config configs/xlsr_local.yaml --folds 0 1 2 3 4
echo "$(date) resume chain complete"