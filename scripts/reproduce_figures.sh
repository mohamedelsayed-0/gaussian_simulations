#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/cascade_figures.py
python scripts/finite_squeezing_span_limit.py
python scripts/ordering_effect.py
python scripts/stochastic_outage.py
python scripts/asymmetric_cascade.py
python scripts/operational_rates.py
