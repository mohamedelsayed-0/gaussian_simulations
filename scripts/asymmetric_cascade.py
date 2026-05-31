#!/usr/bin/env python3
"""Validate the closed asymmetric cascade recurrence."""

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
from cascade_lib import (  # noqa: E402
    amplifier,
    asymmetric_cascade_state,
    nu_minus_asymmetric_cascade_closed,
    nu_minus_exact_asymmetric_cascade,
    pure_loss,
    thermal_loss,
)


FIGDIR = "figs"
DATADIR = "data"


def ensure_dirs() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)


def main() -> None:
    ensure_dirs()
    local_mode = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    transmitted_mode = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.12)]
    r_grid = np.linspace(0.0, 3.0, 300)

    rows: list[dict[str, float]] = []
    closed_vals = []
    exact_vals = []
    for r in r_grid:
        alpha, beta, gamma = asymmetric_cascade_state(float(r), local_mode, transmitted_mode)
        closed = nu_minus_asymmetric_cascade_closed(float(r), local_mode, transmitted_mode)
        exact = nu_minus_exact_asymmetric_cascade(float(r), local_mode, transmitted_mode)
        closed_vals.append(closed)
        exact_vals.append(exact)
        rows.append(
            {
                "r": float(r),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "nu_minus_closed": closed,
                "nu_minus_exact": exact,
                "abs_error": abs(closed - exact),
            }
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(r_grid, closed_vals, label="closed recurrence")
    ax.plot(r_grid, exact_vals, linestyle="--", label="exact CM spectrum")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label=r"$\tilde{\nu}_-=1$")
    ax.set_xlabel("squeezing r")
    ax.set_ylabel(r"$\tilde{\nu}_-$")
    ax.set_title("One-sided asymmetric cascade")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    fig_path = os.path.join(FIGDIR, "asymmetric_cascade.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    csv_path = os.path.join(DATADIR, "asymmetric_cascade.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"max abs error={max(row['abs_error'] for row in rows):.3e}")
    print(f"saved {fig_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
