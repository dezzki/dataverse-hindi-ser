#!/usr/bin/env bash
# DataVerse local GPU training chain: waits for whisper, then trains the big
# SSL models serially (6 GB card can hold one at a time). Launch detached:
#   nohup bash local_train_chain.sh > /tmp/opencode/local_chain.log 2>&1 &
set -u
cd "$(dirname "$0")"
VENV=.venv/bin/python
# order: faster non-aux models first, flagship aux-XLSR last
MODELS=(wavlm_local.yaml hubert_local.yaml xlsr_local.yaml)

echo "$(date) chain start; waiting for whisper-small to finish..."
# wait until whisper run has all 5 folds OR its process has exited & folds done
while :; do
  done_folds=$(ls runs/whisper-small/fold*/best.pt 2>/dev/null | wc -l)
  if [ "$done_folds" -ge 5 ]; then break; fi
  if ! pgrep -f "train_ssl --config configs/whisper_local.yaml" >/dev/null 2>&1; then
    if [ "$done_folds" -ge 5 ]; then break; fi
    echo "$(date) whisper process ended with ${done_folds}/5 folds; proceeding"
    sleep 5
    break
  fi
  sleep 90
done

for m in "${MODELS[@]}"; do
  echo "$(date) === starting $m ==="
  $VENV -m src.train_ssl --config "configs/$m"
  echo "$(date) === done $m ==="
done
echo "$(date) local chain complete"