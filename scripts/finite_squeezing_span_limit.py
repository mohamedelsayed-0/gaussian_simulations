#!/usr/bin/env python3
"""Generate finite-squeezing span-limit data and a manuscript figure."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import n_max_finite_squeezing  # noqa: E402


FIGDIR = "figs"
DATADIR = "data"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def ensure_dirs() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)


def write_csv(rows: list[dict[str, float | int]]) -> str:
    out = os.path.join(DATADIR, "finite_squeezing_span_limit.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["r_max", "eta", "n_max"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot(rows: list[dict[str, float | int]]) -> str:
    etas = sorted({float(row["eta"]) for row in rows})
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for eta in etas:
        subset = [row for row in rows if row["eta"] == eta]
        r_vals = [float(row["r_max"]) for row in subset]
        n_vals = [int(row["n_max"]) for row in subset]
        ax.step(r_vals, n_vals, where="post", label=fr"$\eta={eta}$")

    ax.set_xlabel(r"squeezing budget $r_{\max}$")
    ax.set_ylabel(r"finite-budget span limit $n_{\max}(r_{\max},\eta)$")
    ax.set_title("Finite-squeezing span limit")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "finite_squeezing_span_limit.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def print_latex_table(sample_rows: list[dict[str, float | int]]) -> None:
    print(r"\begin{tabular}{ccc}")
    print(r"\hline")
    print(r"$r_{\max}$ & $\eta$ & $n_{\max}(r_{\max},\eta)$ \\")
    print(r"\hline")
    for row in sample_rows:
        print(f"{row['r_max']:.2f} & {row['eta']:.2f} & {int(row['n_max'])} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def main() -> None:
    ensure_dirs()
    r_grid = np.linspace(0.0, 3.0, 301)
    etas = [0.7, 0.8, 0.9, 0.95, 0.99]
    rows: list[dict[str, float | int]] = []
    for eta in etas:
        for r_max in r_grid:
            rows.append(
                {
                    "r_max": float(r_max),
                    "eta": eta,
                    "n_max": n_max_finite_squeezing(float(r_max), eta),
                }
            )

    csv_path = write_csv(rows)
    fig_path = plot(rows)
    print(f"saved {csv_path}")
    print(f"saved {fig_path}")

    sample_r = [0.5, 1.0, 1.5, 2.0]
    sample_eta = [0.8, 0.9, 0.95, 0.99]
    sample_rows = [
        {"r_max": r_max, "eta": eta, "n_max": n_max_finite_squeezing(r_max, eta)}
        for r_max in sample_r
        for eta in sample_eta
    ]
    print()
    print("LaTeX table")
    print_latex_table(sample_rows)


if __name__ == "__main__":
    main()
