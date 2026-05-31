#!/usr/bin/env python3
"""Operational metrics derived from the scalar cascade value."""

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
from cascade_lib import amplifier, log_negativity, nu_minus_scalar, pure_loss, teleportation_fidelity  # noqa: E402


FIGDIR = "figs"
DATADIR = "data"


def ensure_dirs() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)


def main() -> None:
    ensure_dirs()
    channels = [pure_loss(0.9), amplifier(1.08), pure_loss(0.92)]
    r_grid = np.linspace(0.0, 3.0, 300)
    nu_vals = np.array([nu_minus_scalar(float(r), channels) for r in r_grid])
    logneg = log_negativity(nu_vals)
    fidelity = teleportation_fidelity(nu_vals)

    rows = [
        {
            "r": float(r),
            "nu_minus": float(nu),
            "log_negativity": float(en),
            "teleportation_fidelity": float(fid),
        }
        for r, nu, en, fid in zip(r_grid, nu_vals, logneg, fidelity)
    ]

    fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
    ax1.plot(r_grid, logneg, label="log-negativity", color="tab:blue")
    ax1.set_xlabel("squeezing r")
    ax1.set_ylabel("log-negativity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(r_grid, fidelity, label="teleportation fidelity", color="tab:orange")
    ax2.axhline(0.5, color="black", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("teleportation fidelity", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Operational metrics from the cascade budget")
    fig.tight_layout()

    fig_path = os.path.join(FIGDIR, "operational_rates.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    csv_path = os.path.join(DATADIR, "operational_rates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {fig_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
