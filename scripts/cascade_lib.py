#!/usr/bin/env python3
"""Shared formulas and numerical helpers for Gaussian-channel cascades."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable

import numpy as np


Channel = tuple[float, float]


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


def apply_two_mode_channel(
    V: np.ndarray,
    tau1: float,
    nu1: float,
    tau2: float,
    nu2: float,
) -> np.ndarray:
    a = V[:2, :2]
    c = V[:2, 2:]
    b = V[2:, 2:]
    i2 = np.eye(2)
    return np.block(
        [
            [tau1 * a + nu1 * i2, math.sqrt(tau1 * tau2) * c],
            [math.sqrt(tau1 * tau2) * c.T, tau2 * b + nu2 * i2],
        ]
    )


def partial_transpose_cm(V: np.ndarray) -> np.ndarray:
    p = np.diag([1.0, 1.0, 1.0, -1.0])
    return p @ V @ p


def nu_minus_from_cm(V: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(1j * omega_4() @ partial_transpose_cm(V))
    vals = np.sort(np.abs(eigvals))
    return float(vals[0])


def nu_minus_exact(r: float, channels: Iterable[Channel]) -> float:
    V = tmsv_cm(r)
    for tau, nu in channels:
        V = apply_symmetric_channel(V, tau, nu)
    return nu_minus_from_cm(V)


def nu_minus_exact_asymmetric(
    r: float,
    channel1: Channel,
    channel2: Channel,
) -> float:
    tau1, nu1 = channel1
    tau2, nu2 = channel2
    V = apply_two_mode_channel(tmsv_cm(r), tau1, nu1, tau2, nu2)
    return nu_minus_from_cm(V)


def nu_minus_scalar(r: float, channels: Iterable[Channel]) -> float:
    val = math.exp(-2.0 * r)
    for tau, nu in channels:
        val = tau * val + nu
    return val


def nu_minus_all_orderings(r: float, channels: list[Channel]) -> list[tuple[tuple[Channel, ...], float]]:
    """Compute scalar PT eigenvalues for every distinct channel ordering."""
    seen: set[tuple[Channel, ...]] = set()
    out: list[tuple[tuple[Channel, ...], float]] = []
    for perm in permutations(channels):
        if perm in seen:
            continue
        seen.add(perm)
        out.append((perm, nu_minus_scalar(r, perm)))
    return out


def best_worst_ordering_gap(r: float, channels: list[Channel]) -> tuple[float, float]:
    """Return the best and worst nu_minus values over channel orderings."""
    vals = [value for _, value in nu_minus_all_orderings(r, channels)]
    return min(vals), max(vals)


def nu_minus_asymmetric_closed(r: float, channel1: Channel, channel2: Channel) -> float:
    tau1, nu1 = channel1
    tau2, nu2 = channel2
    a = math.cosh(2.0 * r)
    c = math.sinh(2.0 * r)
    alpha = tau1 * a + nu1
    beta = tau2 * a + nu2
    gamma = math.sqrt(tau1 * tau2) * c
    radicand = (alpha - beta) ** 2 + 4.0 * gamma**2
    nu_sq = 0.5 * (
        alpha**2
        + beta**2
        + 2.0 * gamma**2
        - (alpha + beta) * math.sqrt(radicand)
    )
    return math.sqrt(max(nu_sq, 0.0))


def thermal_loss(eta: float, nth: float) -> Channel:
    return eta, (1.0 - eta) * (2.0 * nth + 1.0)


def pure_loss(eta: float) -> Channel:
    return eta, 1.0 - eta


def amplifier(g: float) -> Channel:
    return g, g - 1.0


def n_amp_nu_minus(r: np.ndarray | float, n: int, g: float) -> np.ndarray | float:
    return (g**n) * np.exp(-2.0 * r) + (g**n - 1.0)


def n_amp_rc(n: int, g: float) -> float:
    gn = g**n
    if gn >= 2.0:
        return math.inf
    return 0.5 * math.log(gn / (2.0 - gn))


def span_boundary(eta: float) -> float:
    return eta / (2.0 * (1.0 - eta))


def n_max_strict(eta: float) -> int:
    boundary = span_boundary(eta)
    nearest = round(boundary)
    if abs(boundary - nearest) < 1e-12:
        return max(0, nearest - 1)
    return max(0, math.floor(boundary))


def n_max_finite_squeezing(r_max: float, eta: float) -> int:
    boundary = eta * (1.0 - math.exp(-2.0 * r_max)) / (2.0 * (1.0 - eta))
    nearest = round(boundary)
    if abs(boundary - nearest) < 1e-12:
        return max(0, nearest - 1)
    return max(0, math.floor(boundary))


def span_rc(n: np.ndarray, eta: float) -> np.ndarray:
    budget = 1.0 - 2.0 * n * (1.0 - eta) / eta
    out = np.full_like(n, np.nan, dtype=float)
    mask = budget > 0.0
    out[mask] = 0.5 * np.log(1.0 / budget[mask])
    return out


def thermal_boundary(eta: float) -> float:
    return eta / (2.0 * (1.0 - eta))


def thermal_sensitivity(eta: float, nth: np.ndarray | float) -> np.ndarray | float:
    denom = 1.0 - (1.0 - eta) * (2.0 * nth + 1.0)
    return (1.0 - eta) / denom


def rc_thermal(eta: float, nth: complex) -> complex:
    denom = 1.0 - (1.0 - eta) * (2.0 * nth + 1.0)
    return 0.5 * np.log(eta / denom)
