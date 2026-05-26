#!/usr/bin/env python3
"""Print span-limit tables for the manuscript appendix."""

from __future__ import annotations

import math


ETAS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]


def db_per_span(eta: float) -> float:
    return -10.0 * math.log10(eta)


def n_max_strict(eta: float) -> int:
    """Largest integer n satisfying n < eta / (2*(1-eta))."""
    boundary = eta / (2.0 * (1.0 - eta))
    nearest = round(boundary)
    if abs(boundary - nearest) < 1e-12:
        return max(0, nearest - 1)
    return max(0, math.floor(boundary))


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
