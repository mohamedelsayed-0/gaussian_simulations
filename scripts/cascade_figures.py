#!/usr/bin/env python3
"""Generate publication figures for cascade entanglement thresholds."""

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
from cascade_lib import (  # noqa: E402
    n_amp_nu_minus,
    n_amp_rc,
    n_max_strict,
    span_boundary,
    span_rc,
    thermal_boundary,
    thermal_sensitivity,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


FIGDIR = "figs"
DATADIR = "data"


def ensure_dirs() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)


def write_rows(filename: str, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> str:
    out = os.path.join(DATADIR, filename)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot_n_amplifier_cascade() -> str:
    r = np.linspace(0.0, 3.0, 500)
    g = 1.3
    rows: list[dict[str, float | int | str]] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for n in range(1, 6):
        cutoff = 2.0 ** (1.0 / n)
        y = n_amp_nu_minus(r, n, g)
        rc = n_amp_rc(n, g)
        for rv, yv in zip(r, y):
            rows.append(
                {
                    "r": float(rv),
                    "n": n,
                    "g": g,
                    "nu_minus": float(yv),
                    "cutoff_gain": cutoff,
                    "r_c": rc if math.isfinite(rc) else "",
                }
            )
        label = f"n={n}, cutoff={cutoff:.3f}"
        ax.plot(r, y, label=label)
        if math.isfinite(rc) and r[0] <= rc <= r[-1]:
            ax.axvline(rc, linestyle=":", linewidth=1.0, alpha=0.75)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=r"$\tilde{\nu}_-=1$")
    ax.set_xlabel("squeezing r")
    ax.set_ylabel(r"$\tilde{\nu}_-$")
    ax.set_title("Identical amplifier cascade, g=1.3")
    ax.set_ylim(0.0, 4.0)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "n_amplifier_cascade.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    write_rows("n_amplifier_cascade.csv", ["r", "n", "g", "nu_minus", "cutoff_gain", "r_c"], rows)
    return out


def plot_amplifier_cutoff_vs_n() -> str:
    ns = np.arange(1, 31)
    cutoffs = 2.0 ** (1.0 / ns)
    marks = [1, 2, 5, 10, 20]
    rows = [{"n": int(n), "cutoff_gain": float(c)} for n, c in zip(ns, cutoffs)]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(ns, cutoffs, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        22.0,
        1.018,
        "asymptote g=1",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )

    for n in marks:
        y = 2.0 ** (1.0 / n)
        ax.scatter([n], [y], color="black", s=22, zorder=3)
        ax.annotate(
            f"n={n}\n{y:.3f}",
            xy=(n, y),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("number of amplifiers n")
    ax.set_ylabel(r"hard cutoff $2^{1/n}$")
    ax.set_title("Per-amplifier gain cutoff")
    ax.set_xlim(1, 30)
    ax.set_ylim(0.98, 2.05)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "amplifier_cutoff_vs_n.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    write_rows("amplifier_cutoff_vs_n.csv", ["n", "cutoff_gain"], rows)
    return out


def plot_span_limit() -> str:
    etas = [0.5, 0.7, 0.85, 0.95]
    n = np.linspace(0.0, 10.0, 1200)
    rows: list[dict[str, float | int | str]] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for eta in etas:
        boundary = span_boundary(eta)
        y = span_rc(n, eta)
        nmax = n_max_strict(eta)
        for nv, yv in zip(n, y):
            rows.append(
                {
                    "n": float(nv),
                    "eta": eta,
                    "r_c": "" if np.isnan(yv) else float(yv),
                    "boundary": boundary,
                    "n_max": nmax,
                }
            )
        ax.plot(n, y, label=fr"$\eta={eta}$, boundary={boundary:.2f}")
        if 0.0 <= boundary <= 10.0:
            ax.axvline(boundary, linestyle=":", linewidth=1.0, alpha=0.75)
            ax.annotate(
                fr"$n_{{max}}={n_max_strict(eta)}$",
                xy=(boundary, 0.2),
                xytext=(3, 0),
                textcoords="offset points",
                rotation=90,
                fontsize=8,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            )

    ax.set_xlabel("number of spans n")
    ax.set_ylabel(r"minimum squeezing $r_c(n,\eta)$")
    ax.set_title("Loss-compensating span limit")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 3.2)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "span_limit.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    write_rows("span_limit.csv", ["n", "eta", "r_c", "boundary", "n_max"], rows)
    return out


def plot_thermal_sensitivity() -> str:
    etas = [0.5, 0.7, 0.9]
    rows: list[dict[str, float | str]] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for eta in etas:
        boundary = thermal_boundary(eta)
        nth = np.linspace(0.0, 0.985 * boundary, 500)
        sens = thermal_sensitivity(eta, nth)
        nth_unit = (eta - (1.0 - eta)) / (2.0 * (1.0 - eta))
        for nthv, sensv in zip(nth, sens):
            rows.append(
                {
                    "N_th": float(nthv),
                    "eta": eta,
                    "sensitivity": float(sensv),
                    "boundary_N_th": boundary,
                    "unit_response_N_th": nth_unit if 0.0 <= nth_unit < boundary else "",
                }
            )
        ax.plot(nth, sens, label=fr"$\eta={eta}$")

        if 0.0 <= nth_unit < boundary:
            ax.scatter([nth_unit], [1.0], color="black", s=18, zorder=3)
            ax.annotate(
                f"{nth_unit:.2f}",
                xy=(nth_unit, 1.0),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"thermal photon number $N_{\mathrm{th}}$")
    ax.set_ylabel(r"$\partial r_c / \partial N_{\mathrm{th}}$")
    ax.set_title("Thermal-noise sensitivity")
    ax.set_ylim(0.0, 8.0)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "thermal_sensitivity.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    write_rows(
        "thermal_sensitivity.csv",
        ["N_th", "eta", "sensitivity", "boundary_N_th", "unit_response_N_th"],
        rows,
    )
    return out


def main() -> None:
    ensure_dirs()
    outputs = [
        plot_n_amplifier_cascade(),
        plot_amplifier_cutoff_vs_n(),
        plot_span_limit(),
        plot_thermal_sensitivity(),
    ]
    for out in outputs:
        print(f"saved {out}")


if __name__ == "__main__":
    main()
