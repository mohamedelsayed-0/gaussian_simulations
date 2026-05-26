#!/usr/bin/env python3
"""Validate cascade formulas against exact symplectic eigenvalues."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_lib import (  # noqa: E402
    amplifier,
    nu_minus_exact,
    nu_minus_scalar,
    pure_loss,
    rc_thermal,
    thermal_loss,
    thermal_sensitivity,
)


TOL = 1e-10


@dataclass
class CheckResult:
    name: str
    max_err: float

    @property
    def passed(self) -> bool:
        return self.max_err < TOL


def max_exact_scalar_error(cases: Iterable[tuple[float, list[tuple[float, float]]]]) -> float:
    return max(abs(nu_minus_exact(r, ch) - nu_minus_scalar(r, ch)) for r, ch in cases)


def check_original_thermal() -> CheckResult:
    cases = []
    for r in np.linspace(0.0, 3.0, 31):
        for eta in [0.2, 0.5, 0.8, 0.95]:
            for nth in [0.0, 0.05, 0.2, 0.8]:
                cases.append((float(r), [thermal_loss(eta, nth)]))
    return CheckResult("Original thermal threshold", max_exact_scalar_error(cases))


def check_original_amplifier() -> CheckResult:
    cases = []
    for r in np.linspace(0.0, 3.0, 31):
        for g in [1.05, 1.2, 1.5, 1.95, 2.0, 3.0]:
            cases.append((float(r), [amplifier(g)]))
    return CheckResult("Original amplifier threshold", max_exact_scalar_error(cases))


def check_cascade() -> CheckResult:
    channel_sets = [
        [pure_loss(0.8), thermal_loss(0.7, 0.1), amplifier(1.2)],
        [amplifier(1.1), pure_loss(0.9), amplifier(1.05), thermal_loss(0.85, 0.03)],
        [thermal_loss(0.6, 0.2), thermal_loss(0.75, 0.05), pure_loss(0.95)],
    ]
    cases = [(float(r), channels) for r in np.linspace(0.0, 2.5, 41) for channels in channel_sets]
    return CheckResult("Proposition 4 (cascade)", max_exact_scalar_error(cases))


def check_n_amp() -> CheckResult:
    errors = []
    for r in np.linspace(0.0, 3.0, 31):
        for n in range(1, 8):
            for g in [1.02, 1.1, 1.3, 1.6]:
                channels = [amplifier(g)] * n
                closed = (g**n) * math.exp(-2.0 * float(r)) + (g**n - 1.0)
                exact = nu_minus_exact(float(r), channels)
                errors.append(abs(exact - closed))
    return CheckResult("Proposition 5 (n-amp)", max(errors))


def check_spans() -> CheckResult:
    errors = []
    for r in np.linspace(0.0, 3.0, 31):
        for eta in [0.5, 0.7, 0.85, 0.95, 0.99]:
            span = [pure_loss(eta), amplifier(1.0 / eta)]
            for n in range(1, 10):
                channels = span * n
                closed = math.exp(-2.0 * float(r)) + 2.0 * n * (1.0 - eta) / eta
                exact = nu_minus_exact(float(r), channels)
                errors.append(abs(exact - closed))
    return CheckResult("Proposition 6 (spans)", max(errors))


def check_gain_split() -> CheckResult:
    gain_sets = [
        [1.05, 1.15, 1.2],
        [1.01, 1.04, 1.08, 1.12],
        [1.4, 1.05],
    ]
    errors = []
    for r in np.linspace(0.0, 3.0, 31):
        for gains in gain_sets:
            channels = [amplifier(g) for g in gains]
            total_gain = math.prod(gains)
            closed = total_gain * math.exp(-2.0 * float(r)) + (total_gain - 1.0)
            exact = nu_minus_exact(float(r), channels)
            errors.append(abs(exact - closed))
    return CheckResult("Proposition 7 (gain-split)", max(errors))


def check_sensitivity() -> CheckResult:
    errors = []
    h = 1e-30
    for eta in [0.55, 0.7, 0.9, 0.98]:
        boundary = eta / (2.0 * (1.0 - eta))
        for nth in np.linspace(0.0, 0.9 * boundary, 30):
            complex_deriv = np.imag(rc_thermal(eta, float(nth) + 1j * h)) / h
            closed = thermal_sensitivity(eta, float(nth))
            errors.append(abs(float(complex_deriv) - closed))
    return CheckResult("Proposition 8 (sensitivity)", max(errors))


def run_checks() -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        check_original_thermal,
        check_original_amplifier,
        check_cascade,
        check_n_amp,
        check_spans,
        check_gain_split,
        check_sensitivity,
    ]
    return [check() for check in checks]


def print_results(results: list[CheckResult]) -> None:
    width = max(len(r.name) for r in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name:<{width}}  max err = {result.max_err:.2e}  [{status}]")


def main() -> None:
    results = run_checks()
    print_results(results)
    failed = [r for r in results if not r.passed]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
