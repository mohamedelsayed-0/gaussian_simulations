from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cascade_lib import (  # noqa: E402
    amplifier,
    n_amp_nu_minus,
    nu_minus_asymmetric_closed,
    nu_minus_exact,
    nu_minus_exact_asymmetric,
    nu_minus_scalar,
    pure_loss,
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
            assert abs(exact - closed) < 1e-10
