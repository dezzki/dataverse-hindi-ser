#!/usr/bin/env bash
# Resume after machine-off: pull completed kernel probs, re-push wavlm_aux,
# relaunch watchdog. Run from the repo:  bash scripts/resume_after_off.sh
set -u
cd "$(dirname "$0")/.."
COMP=dataverse-detecting-emotions-from-hindi-speech

echo "[resume] pulling hubert + xlsr_aux OOF/test probs"
for m in hubert xlsr_aux; do
  case $m in
    hubert) tag=hubert-large-ls960-ft ;;
    xlsr_aux) tag=wav2vec2-large-xlsr-53_aux ;;
  esac
  mkdir -p kaggle_dl/$m oof/$tag
  kaggle kernels output dezzki/dataverse-$m -p kaggle_dl/$m --file-pattern '.*probs.*\.csv' -o -q 2>&1 | tail -1
  for f in kaggle_dl/$m/oof_probs_*.csv kaggle_dl/$m/test_probs_*.csv; do [ -f "$f" ] && cp "$f" oof/$tag/; done
  ls oof/$tag 2>/dev/null | head
done

echo "[resume] re-pushing wavlm_aux (GPU slot should be free)"
kaggle kernels push -p kernels/wavlm_aux 2>&1 | grep -iE "successfully|error|reached" | tail -1

echo "[resume] relaunching watchdog (native-background friendly)"
pkill -f auto_pipeline.py 2>/dev/null
nohup env PYTHONUNBUFFERED=1 .venv/bin/python auto_pipeline.py --interval 300 > /tmp/opencode/auto_pipeline.log 2>&1 &
echo "[resume] watchdog pid $!"
echo "[resume] done. Consider also relaunching local training if you want local compute."