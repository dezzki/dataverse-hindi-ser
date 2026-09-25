#!/usr/bin/env bash
# Auto-push remaining Kaggle GPU kernels as slots free (2-concurrent limit).
# Pushes: hubert, then xlsr_aux, then wavlm_aux — each when a GPU slot opens.
set -u
cd "$(dirname "$0")"
PENDING=(hubert xlsr_aux wavlm_aux)
FREE=()
RUNNING_SLOTS=(dezzki/dataverse-xlsr dezzki/dataverse-wavlm)

log(){ echo "$(date) $*"; }

# wait until a tracked kernel is no longer running/queued (slot free)
wait_slot(){
  while :; do
    for slug in "${RUNNING_SLOTS[@]}"; do
      st=$(kaggle kernels status "$slug" 2>&1 | grep -o 'status "[^"]*"' | sed 's/status "//; s/"//')
      case "$st" in *COMPLETE*|*ERROR*|*CANCELED*|*CANCELLED*) echo "$slug $st" ;; esac
    done
    sleep 120
  done
}

for m in "${PENDING[@]}"; do
  log "waiting for a free GPU slot before pushing $m..."
  # a slot is free when any current/held kernel is not RUNNING/QUEUED/PENDING
  free_found=""
  for _ in $(seq 1 200); do
    for slug in "${RUNNING_SLOTS[@]}"; do
      st=$(kaggle kernels status "$slug" 2>&1 | grep -o 'status "[^"]*"' | sed 's/status "//; s/"//')
      case "$st" in
        COMPLETE|completed|ERROR|errored|CANCELED|CANCELLED) free_found=1 ;;
      esac
    done
    [ -n "$free_found" ] && break
    sleep 60
  done
  if [ -z "$free_found" ]; then
    log "no slot freed in time for $m; trying push anyway"
  fi
  log "pushing $m"
  kaggle kernels push -p "kernels/$m" 2>&1 | tail -1
  # mark it as a slot holder so we wait on it next
  RUNNING_SLOTS+=("dezzki/dataverse-$m")
  sleep 30
done
log "all queued kernels pushed"