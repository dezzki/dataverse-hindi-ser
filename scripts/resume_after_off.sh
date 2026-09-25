#!/usr/bin/env bash
# Resume after machine-off: pull completed kernel probs -> oof/, re-push any
# pending GPU kernels, re-ensemble + submit, relaunch watchdog.
# Run: bash scripts/resume_after_off.sh
set -u
cd "$(dirname "$0")/.."
COMP=dataverse-detecting-emotions-from-hindi-speech
# slug -> tag (hyphen slugs are the real kernel URLs)
declare -A KERN=(
  [hubert]=hubert-large-ls960-ft
  [xlsr_aux]=wav2vec2-large-xlsr-53_aux
  [wavlm_aux]=wavlm-large_aux
  [hubert_aux]=hubert-large-ls960-ft_aux
)

echo "[resume] pulling completed kernel prob outputs"
for slug in "${!KERN[@]}"; do
  tag="${KERN[$slug]}"
  hslug="${slug//_/-}"
  mkdir -p "kaggle_dl/$slug" "oof/$tag"
  kaggle kernels output "dezzki/dataverse-$hslug" -p "kaggle_dl/$slug" --file-pattern '.*probs.*\.csv' -o -q 2>&1 | tail -1
  for f in "kaggle_dl/$slug/oof/$tag/"oof_probs_*.csv "kaggle_dl/$slug/oof/$tag/"test_probs_*.csv; do
    [ -f "$f" ] && cp "$f" "oof/$tag/"
  done
done

echo "[resume] re-pushing any kernels not yet RUNNING/COMPLETE"
for slug in hubert xlsr_aux wavlm_aux hubert_aux; do
  hslug="${slug//_/-}"
  st=$(kaggle kernels status "dezzki/dataverse-$hslug" 2>&1 | grep -o 'status "[^"]*"' | sed 's/status "//; s/"//')
  echo "  $hslug: $st"
  case "$st" in
    ''|*ERROR*) kaggle kernels push -p "kernels/$slug" 2>&1 | grep -iE "successfully|error|reached" | tail -1 ;;
  esac
done

echo "[resume] re-ensemble all available OOF + submit"
source .venv/bin/activate
python -m src.ensemble --tags classic wav2vec2-base-superb-er whisper-small wav2vec2-large-xlsr-53 wavlm-large hubert-large-ls960-ft wav2vec2-large-xlsr-53_aux "$@" --out runs/final_resume 2>&1 | grep -iE "blend OOF|calibrated OOF" | tail -2
python -m src.submit --probs runs/final_resume/test_probs.csv --data-root data/SRCASW-BhavVaani --out runs/final_resume/submission.csv 2>&1 | tail -2
n=$(kaggle competitions submission-limits -c "$COMP" 2>&1 | grep -o 'Remaining today: [0-9]*' | head -1)
echo "submissions: $n"
if [ "${n##* }" -gt 0 ]; then
  kaggle competitions submit -c "$COMP" -f runs/final_resume/submission.csv -m "resume ensemble" 2>&1 | grep -iE "successfully" | tail -1
fi

echo "[resume] relaunching watchdog"
pkill -f auto_pipeline.py 2>/dev/null
nohup env PYTHONUNBUFFERED=1 .venv/bin/python auto_pipeline.py --interval 300 > /tmp/opencode/auto_pipeline.log 2>&1 &
echo "[resume] watchdog pid $!"