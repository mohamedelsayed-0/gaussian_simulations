from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cascade_lib import (  # noqa: E402
    amplifier,
    beta_span_noise_moments,
    loss_compensating_survives,
    n_max_finite_squeezing,
    n_max_strict,
    nu_minus_exact,
    outage_survival_clt,
    pure_loss,
    rc_thermal,
    required_squeezing_for_reliability_db,
    span_boundary,
    thermal_loss,
    thermal_sensitivity,
)


def test_thermal_and_amplifier_thresholds_match_exact_values() -> None:
    for r in np.linspace(0.0, 3.0, 21):
        for eta in [0.2, 0.5, 0.8, 0.95]:
            for nth in [0.0, 0.05, 0.2, 0.8]:
                exact = nu_minus_exact(float(r), [thermal_loss(eta, nth)])
                closed = eta * math.exp(-2.0 * float(r)) + (1.0 - eta) * (2.0 * nth + 1.0)
                assert abs(exact - closed) < 1e-10

        for g in [1.05, 1.2, 1.5, 1.95, 2.0, 3.0]:
            exact = nu_minus_exact(float(r), [amplifier(g)])
            closed = g * math.exp(-2.0 * float(r)) + (g - 1.0)
            assert abs(exact - closed) < 1e-10


def test_span_limit_excludes_exact_boundary_cases() -> None:
    assert span_boundary(0.80) == 2.0000000000000004 or math.isclose(span_boundary(0.80), 2.0)
    assert n_max_strict(0.80) == 1
    assert n_max_strict(0.50) == 0
    assert n_max_strict(0.95) == 9


def test_finite_squeezing_span_limit_is_bounded_by_infinite_limit() -> None:
    for eta in [0.7, 0.8, 0.9, 0.95, 0.99]:
        infinite_limit = n_max_strict(eta)
        for r_max in [0.0, 0.5, 1.0, 2.0, 3.0]:
            finite_limit = n_max_finite_squeezing(r_max, eta)
            assert 0 <= finite_limit <= infinite_limit


def test_thermal_sensitivity_matches_complex_step_derivative() -> None:
    h = 1e-30
    for eta in [0.55, 0.7, 0.9, 0.98]:
        boundary = eta / (2.0 * (1.0 - eta))
        for nth in np.linspace(0.0, 0.9 * boundary, 10):
            complex_deriv = np.imag(rc_thermal(eta, float(nth) + 1j * h)) / h
            closed = thermal_sensitivity(eta, float(nth))
            assert abs(float(complex_deriv) - closed) < 1e-10


def test_random_span_outage_budget_uses_loss_compensating_sum() -> None:
    etas = [0.98, 0.97, 0.99]
    assert loss_compensating_survives(1.0, etas)
    assert not loss_compensating_survives(0.01, etas)


def test_beta_span_noise_moments_and_required_squeezing_are_finite() -> None:
    mean, variance = beta_span_noise_moments(120.0, 6.0)
    assert mean > 0.0
    assert variance > 0.0
    prob = outage_survival_clt(1.0, 5, 120.0, 6.0)
    required_db = required_squeezing_for_reliability_db(5, 120.0, 6.0, 0.95)
    assert 0.0 <= prob <= 1.0
    assert math.isfinite(required_db)
    assert required_db > 0.0
