#!/usr/bin/env python3
"""Validate the exact asymmetric two-channel PT eigenvalue formula."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import (  # noqa: E402
    amplifier,
    nu_minus_asymmetric_closed,
    nu_minus_exact_asymmetric,
    pure_loss,
    thermal_loss,
)


TOL = 1e-10


def main() -> None:
    channels = [
        (pure_loss(0.8), pure_loss(0.6)),
        (thermal_loss(0.9, 0.05), thermal_loss(0.7, 0.15)),
        (amplifier(1.1), amplifier(1.4)),
        (pure_loss(0.85), amplifier(1.2)),
        ((1.0, 0.0), thermal_loss(0.75, 0.1)),
    ]
    errors = []
    worst = None
    for r in np.linspace(0.0, 3.0, 61):
        for ch1, ch2 in channels:
            exact = nu_minus_exact_asymmetric(float(r), ch1, ch2)
            closed = nu_minus_asymmetric_closed(float(r), ch1, ch2)
            err = abs(exact - closed)
            errors.append(err)
            if worst is None or err > worst[0]:
                worst = (err, float(r), ch1, ch2, exact, closed)

    max_err = max(errors)
    status = "PASS" if max_err < TOL else "FAIL"
    print(f"Asymmetric two-channel formula  max err = {max_err:.2e}  [{status}]")
    if status == "FAIL" and worst is not None:
        err, r, ch1, ch2, exact, closed = worst
        print(f"worst r={r}, channel1={ch1}, channel2={ch2}, exact={exact}, closed={closed}, err={err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
