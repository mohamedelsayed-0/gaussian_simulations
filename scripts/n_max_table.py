#!/usr/bin/env python3
"""Print span-limit tables for the manuscript appendix."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import n_max_strict  # noqa: E402


ETAS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]


def db_per_span(eta: float) -> float:
    return -10.0 * math.log10(eta)


def print_plain_text() -> None:
    print("Plain-text table")
    print("eta     dB/span    n_max")
    for eta in ETAS:
        note = "  (no span survives)" if n_max_strict(eta) == 0 else ""
        print(f"{eta:0.2f}    {db_per_span(eta):0.2f} dB    {n_max_strict(eta):>2}{note}")


def print_latex() -> None:
    print()
    print("LaTeX table")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{ccc}")
    print(r"\hline")
    print(r"$\eta$ & dB/span & $n_{\max}$ \\")
    print(r"\hline")
    for eta in ETAS:
        note = r" \;(\mathrm{no\ span})" if n_max_strict(eta) == 0 else ""
        print(f"{eta:0.2f} & {db_per_span(eta):0.2f} & {n_max_strict(eta)}{note} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Strict span limit for loss-compensating spans. Entanglement requires $n < \eta/[2(1-\eta)]$, so exact-boundary cases are not counted as surviving spans.}")
    print(r"\end{table}")


def main() -> None:
    print_plain_text()
    print_latex()


if __name__ == "__main__":
    main()
