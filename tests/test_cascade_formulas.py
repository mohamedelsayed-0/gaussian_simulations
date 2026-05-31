from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cascade_lib import (  # noqa: E402
    accumulated_noise,
    amplifier,
    log_negativity,
    best_worst_ordering_gap,
    n_amp_nu_minus,
    nu_minus_all_orderings,
    nu_minus_asymmetric_cascade_closed,
    nu_minus_asymmetric_closed,
    nu_minus_exact,
    nu_minus_exact_asymmetric_cascade,
    nu_minus_exact_asymmetric,
    nu_minus_scalar,
    optimal_ordering,
    pure_loss,
    teleportation_fidelity,
    thermal_loss,
)


def test_general_cascade_matches_exact_symplectic_eigenvalue() -> None:
    channel_sets = [
        [pure_loss(0.8), thermal_loss(0.7, 0.1), amplifier(1.2)],
        [amplifier(1.1), pure_loss(0.9), amplifier(1.05), thermal_loss(0.85, 0.03)],
        [thermal_loss(0.6, 0.2), thermal_loss(0.75, 0.05), pure_loss(0.95)],
    ]
    for r in np.linspace(0.0, 2.5, 21):
        for channels in channel_sets:
            exact = nu_minus_exact(float(r), channels)
            scalar = nu_minus_scalar(float(r), channels)
            assert abs(exact - scalar) < 1e-10


def test_n_amplifier_closed_form_matches_exact() -> None:
    for r in np.linspace(0.0, 3.0, 21):
        for n in range(1, 7):
            for g in [1.02, 1.1, 1.3, 1.6]:
                channels = [amplifier(g)] * n
                exact = nu_minus_exact(float(r), channels)
                closed = float(n_amp_nu_minus(float(r), n, g))
                assert abs(exact - closed) < 1e-10


def test_gain_split_depends_only_on_total_gain_without_loss() -> None:
    gain_sets = [
        [1.05, 1.15, 1.2],
        [1.01, 1.04, 1.08, 1.12],
        [1.4, 1.05],
    ]
    for r in np.linspace(0.0, 3.0, 21):
        for gains in gain_sets:
            channels = [amplifier(g) for g in gains]
            total_gain = math.prod(gains)
            exact = nu_minus_exact(float(r), channels)
            closed = total_gain * math.exp(-2.0 * float(r)) + (total_gain - 1.0)
            assert abs(exact - closed) < 1e-10


def test_pure_amplifier_ordering_is_invariant() -> None:
    channels = [amplifier(1.04), amplifier(1.11), amplifier(1.23)]
    for r in np.linspace(0.0, 3.0, 11):
        values = [value for _, value in nu_minus_all_orderings(float(r), channels)]
        assert max(values) - min(values) < 1e-12


def test_mixed_channel_ordering_can_change_survival_margin() -> None:
    channels = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.25)]
    best, worst = best_worst_ordering_gap(1.0, channels)
    assert best < worst
    assert worst - best > 0.05


def test_optimal_ordering_matches_bruteforce_noise_minimum() -> None:
    channels = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.25)]
    ordered = optimal_ordering(channels)
    brute_min = min(accumulated_noise(order) for order, _ in nu_minus_all_orderings(1.0, channels))
    assert accumulated_noise(ordered) == brute_min
    assert ordered == [amplifier(1.25), thermal_loss(0.9, 0.08), pure_loss(0.82)]


def test_asymmetric_formula_matches_exact_symplectic_eigenvalue() -> None:
    channel_pairs = [
        (pure_loss(0.8), pure_loss(0.6)),
        (thermal_loss(0.9, 0.05), thermal_loss(0.7, 0.15)),
        (amplifier(1.1), amplifier(1.4)),
        (pure_loss(0.85), amplifier(1.2)),
        ((1.0, 0.0), thermal_loss(0.75, 0.1)),
    ]
    for r in np.linspace(0.0, 3.0, 21):
        for channel1, channel2 in channel_pairs:
            exact = nu_minus_exact_asymmetric(float(r), channel1, channel2)
            closed = nu_minus_asymmetric_closed(float(r), channel1, channel2)
            assert abs(exact - closed) < 1e-11


def test_asymmetric_cascade_recurrence_matches_exact_spectrum() -> None:
    mode1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    mode2 = [pure_loss(0.82), thermal_loss(0.9, 0.08), amplifier(1.12)]
    for r in np.linspace(0.0, 3.0, 21):
        exact = nu_minus_exact_asymmetric_cascade(float(r), mode1, mode2)
        closed = nu_minus_asymmetric_cascade_closed(float(r), mode1, mode2)
        assert abs(exact - closed) < 1e-11


def test_operational_metrics_are_monotone_functions_of_nu_minus() -> None:
    nu_values = np.array([0.25, 0.5, 1.0, 1.5])
    logneg = log_negativity(nu_values)
    fidelity = teleportation_fidelity(nu_values)
    assert np.all(np.diff(logneg) <= 0.0)
    assert np.all(np.diff(fidelity) < 0.0)
    assert logneg[-1] == 0.0
    assert fidelity[2] == 0.5
