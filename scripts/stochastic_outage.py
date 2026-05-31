#!/usr/bin/env python3
"""Stochastic loss-compensating span outage budgets."""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import outage_survival_clt, required_squeezing_for_reliability_db  # noqa: E402


FIGDIR = "figs"
DATADIR = "data"


def ensure_dirs() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)


def main() -> None:
    ensure_dirs()
    alpha = 120.0
    beta = 6.0
    reliability = 0.95
    spans = np.arange(1, 41)
    r_grid = np.linspace(0.05, 2.5, 120)

    rows: list[dict[str, float | int | str]] = []
    for n in spans:
        required_db = required_squeezing_for_reliability_db(int(n), alpha, beta, reliability)
        rows.append(
            {
                "series": "required_squeezing",
                "n": int(n),
                "r": "",
                "survival_probability": reliability,
                "required_squeezing_db": required_db if math.isfinite(required_db) else "",
                "beta_alpha": alpha,
                "beta_beta": beta,
            }
        )

    selected_spans = [5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for n in selected_spans:
        probs = [outage_survival_clt(float(r), n, alpha, beta) for r in r_grid]
        for r, prob in zip(r_grid, probs):
            rows.append(
                {
                    "series": f"survival_n_{n}",
                    "n": n,
                    "r": float(r),
                    "survival_probability": float(prob),
                    "required_squeezing_db": "",
                    "beta_alpha": alpha,
                    "beta_beta": beta,
                }
            )
        ax.plot(20.0 * r_grid * math.log10(math.e), probs, label=f"n={n}")

    ax.axhline(reliability, color="black", linestyle="--", linewidth=1.0, label="95%")
    ax.set_xlabel("input squeezing (dB)")
    ax.set_ylabel("CLT survival probability")
    ax.set_title("Entanglement outage for random span loss")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    fig_path = os.path.join(FIGDIR, "stochastic_outage.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    csv_path = os.path.join(DATADIR, "stochastic_outage.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {fig_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
