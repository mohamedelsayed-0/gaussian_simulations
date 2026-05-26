#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python benchmark-simulation/gaussian_en_bound.py
python benchmark-simulation/non_tmv.py
python scripts/cascade_figures.py
python scripts/finite_squeezing_span_limit.py
