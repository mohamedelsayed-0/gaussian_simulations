#!/usr/bin/env python3
"""Show how channel ordering changes entanglement survival in mixed cascades."""

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
    Channel,
    amplifier,
    best_worst_ordering_gap,
    nu_minus_all_orderings,
    pure_loss,
    thermal_loss,
)


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


def channel_name(channel: Channel) -> str:
    if channel == pure_loss(0.82):
        return "loss"
    if channel == thermal_loss(0.9, 0.08):
        return "thermal"
    if channel == amplifier(1.25):
        return "amp"
    return f"({channel[0]:.3g},{channel[1]:.3g})"


def order_name(ordering: tuple[Channel, ...]) -> str:
    return " -> ".join(channel_name(ch) for ch in ordering)


def write_csv(rows: list[dict[str, float | str]]) -> str:
    out = os.path.join(DATADIR, "ordering_effect_gap.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["r", "series", "nu_minus", "ordering"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot_ordering_effect() -> tuple[str, str]:
    channels = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.25)]
    r_grid = np.linspace(0.0, 3.0, 500)

    best_vals = []
    worst_vals = []
    median_vals = []
    best_order = None
    worst_order = None
    rows: list[dict[str, float | str]] = []

    for r in r_grid:
        ordered = sorted(nu_minus_all_orderings(float(r), channels), key=lambda item: item[1])
        best_order, best = ordered[0]
        worst_order, worst = ordered[-1]
        median_order, median = ordered[len(ordered) // 2]
        best_vals.append(best)
        worst_vals.append(worst)
        median_vals.append(median)
        rows.extend(
            [
                {"r": float(r), "series": "best", "nu_minus": best, "ordering": order_name(best_order)},
                {"r": float(r), "series": "median", "nu_minus": median, "ordering": order_name(median_order)},
                {"r": float(r), "series": "worst", "nu_minus": worst, "ordering": order_name(worst_order)},
            ]
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(r_grid, best_vals, label=f"best: {order_name(best_order)}")
    ax.plot(r_grid, median_vals, label="middle ordering", linestyle="-.")
    ax.plot(r_grid, worst_vals, label=f"worst: {order_name(worst_order)}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=r"$\tilde{\nu}_-=1$")
    ax.fill_between(r_grid, best_vals, worst_vals, color="gray", alpha=0.14, label="ordering gap")
    ax.set_xlabel("squeezing r")
    ax.set_ylabel(r"$\tilde{\nu}_-$")
    ax.set_title("Ordering effect in a mixed channel cascade")
    ax.set_ylim(0.0, max(2.4, float(np.nanmax(worst_vals)) * 1.05))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    fig_path = os.path.join(FIGDIR, "ordering_effect_gap.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    csv_path = write_csv(rows)
    return fig_path, csv_path


def main() -> None:
    ensure_dirs()
    fig_path, csv_path = plot_ordering_effect()
    print(f"saved {fig_path}")
    print(f"saved {csv_path}")

    channels = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.25)]
    best, worst = best_worst_ordering_gap(1.0, channels)
    print(f"at r=1.0: best nu_minus={best:.6f}, worst nu_minus={worst:.6f}")


if __name__ == "__main__":
    main()
