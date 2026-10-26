#!/usr/bin/env bash
# Runs the whole study from scratch (about 6 to 8 hours on a 6-core CPU).
# Every script is resumable: if it is interrupted, run it again and it continues.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

$PY scripts/00_explore.py
$PY scripts/01_train_baselines.py 2>&1 | tee logs/01_baselines.log
$PY scripts/02_quantize.py        2>&1 | tee logs/02_quantize.log
$PY scripts/03_prune.py           2>&1 | tee logs/03_prune.log
$PY scripts/04_distill.py         2>&1 | tee logs/04_distill.log
$PY scripts/05_robustness.py      2>&1 | tee logs/05_robustness.log
# latency must be measured with nothing else running
$PY scripts/06_results.py         2>&1 | tee logs/06_results.log
