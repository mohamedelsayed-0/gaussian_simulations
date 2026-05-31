#!/usr/bin/env python3
"""Validate and export the optimal-ordering design rule."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import (  # noqa: E402
    Channel,
    accumulated_noise,
    amplifier,
    optimal_ordering,
    ordering_score,
    pure_loss,
    thermal_loss,
)


DATADIR = "data"


def channel_label(channel: Channel) -> str:
    labels = {
        amplifier(1.25): "amplifier g=1.25",
        thermal_loss(0.9, 0.08): "thermal loss eta=0.90 Nth=0.08",
        pure_loss(0.82): "pure loss eta=0.82",
    }
    return labels.get(channel, f"tau={channel[0]:.4g}, noise={channel[1]:.4g}")


def main() -> None:
    os.makedirs(DATADIR, exist_ok=True)
    channels = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.25)]
    ordered = optimal_ordering(channels)

    rows = [
        {
            "rank": rank,
            "channel": channel_label(channel),
            "tau": channel[0],
            "noise": channel[1],
            "ordering_score": ordering_score(channel),
        }
        for rank, channel in enumerate(ordered, start=1)
    ]

    out = os.path.join(DATADIR, "optimal_ordering.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("optimal ordering:")
    for row in rows:
        print(f"  {row['rank']}. {row['channel']} (score={row['ordering_score']:.6g})")
    print(f"accumulated noise={accumulated_noise(ordered):.6f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
