#!/usr/bin/env bash
# DataVerse auto-pipeline launcher (unattended watchdog).
set -euo pipefail
cd "$(dirname "$0")"
exec env PYTHONUNBUFFERED=1 .venv/bin/python auto_pipeline.py "$@"