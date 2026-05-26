#!/usr/bin/env python3
"""Validate cascade formulas against exact symplectic eigenvalues."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


TOL = 1e-10


@dataclass
class CheckResult:
    name: str
    max_err: float

    @property
    def passed(self) -> bool:
        return self.max_err < TOL


def omega_2() -> np.ndarray:
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


def omega_4() -> np.ndarray:
    o2 = omega_2()
    z2 = np.zeros((2, 2))
    return np.block([[o2, z2], [z2, o2]])


def tmsv_cm(r: float) -> np.ndarray:
    a = math.cosh(2.0 * r)
    c = math.sinh(2.0 * r)
    i2 = np.eye(2)
    z = np.diag([1.0, -1.0])
    return np.block([[a * i2, c * z], [c * z, a * i2]])


def apply_symmetric_channel(V: np.ndarray, tau: float, nu: float) -> np.ndarray:
    a = V[:2, :2]
    c = V[:2, 2:]
    b = V[2:, 2:]
    i2 = np.eye(2)
    return np.block(
        [
            [tau * a + nu * i2, tau * c],
            [tau * c.T, tau * b + nu * i2],
        ]
    )


def nu_minus_exact(r: float, channels: Iterable[tuple[float, float]]) -> float:
    V = tmsv_cm(r)
    for tau, nu in channels:
        V = apply_symmetric_channel(V, tau, nu)

    p = np.diag([1.0, 1.0, 1.0, -1.0])
    vpt = p @ V @ p
    eigvals = np.linalg.eigvals(1j * omega_4() @ vpt)
    vals = np.sort(np.abs(eigvals.real + 1j * eigvals.imag))
    return float(vals[0])


def nu_minus_scalar(r: float, channels: Iterable[tuple[float, float]]) -> float:
    val = math.exp(-2.0 * r)
    for tau, nu in channels:
        val = tau * val + nu
    return val


def thermal_loss(eta: float, nth: float) -> tuple[float, float]:
    return eta, (1.0 - eta) * (2.0 * nth + 1.0)


def pure_loss(eta: float) -> tuple[float, float]:
    return eta, 1.0 - eta


def amplifier(g: float) -> tuple[float, float]:
    return g, g - 1.0


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
    cases = []
    for r in np.linspace(0.0, 3.0, 31):
        for n in range(1, 8):
            for g in [1.02, 1.1, 1.3, 1.6]:
                channels = [amplifier(g)] * n
                closed = (g**n) * math.exp(-2.0 * float(r)) + (g**n - 1.0)
                exact = nu_minus_exact(float(r), channels)
                cases.append(abs(exact - closed))
    return CheckResult("Proposition 5 (n-amp)", max(cases))


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
            G = math.prod(gains)
            closed = G * math.exp(-2.0 * float(r)) + (G - 1.0)
            exact = nu_minus_exact(float(r), channels)
            errors.append(abs(exact - closed))
    return CheckResult("Proposition 7 (gain-split)", max(errors))


def rc_thermal(eta: float, nth: complex) -> complex:
    denom = 1.0 - (1.0 - eta) * (2.0 * nth + 1.0)
    return 0.5 * np.log(eta / denom)


def thermal_sensitivity(eta: float, nth: float) -> float:
    return (1.0 - eta) / (1.0 - (1.0 - eta) * (2.0 * nth + 1.0))


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


def print_results(results: list[CheckResult]) -> None:
    width = max(len(r.name) for r in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name:<{width}}  max err = {result.max_err:.2e}  [{status}]")


def main() -> None:
    checks: list[Callable[[], CheckResult]] = [
        check_original_thermal,
        check_original_amplifier,
        check_cascade,
        check_n_amp,
        check_spans,
        check_gain_split,
        check_sensitivity,
    ]
    results = [check() for check in checks]
    print_results(results)
    failed = [r for r in results if not r.passed]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
