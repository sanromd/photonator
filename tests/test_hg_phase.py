"""Tests for Henyey-Greenstein phase function sampling."""

import numpy as np
import pytest
from scipy.stats import kstest  # type: ignore[import]

from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction


@pytest.mark.parametrize("g", [0.0, 0.5, 0.9, -0.5])
def test_hg_sample_range(g: float) -> None:
    """Sampled angles must lie in [0, π]."""
    rng = np.random.default_rng(0)
    pf = HenyeyGreensteinPhaseFunction(g=g)
    theta = pf.sample(10_000, rng)
    assert np.all(theta >= 0.0), "Negative angles found"
    assert np.all(theta <= np.pi), "Angles > π found"


def test_hg_isotropic_uniform_cos() -> None:
    """g=0 → cos(theta) must be uniform on [-1, 1] (KS test)."""
    rng = np.random.default_rng(42)
    pf = HenyeyGreensteinPhaseFunction(g=0.0)
    theta = pf.sample(50_000, rng)
    cos_theta = np.cos(theta)
    # cos_theta ~ Uniform(-1, 1)  →  (cos_theta + 1)/2 ~ Uniform(0, 1)
    stat, p = kstest((cos_theta + 1.0) / 2.0, "uniform")
    assert p > 0.05, f"KS test failed for isotropic HG: p={p:.4f}"


@pytest.mark.parametrize("g", [0.5, 0.9])
def test_hg_forward_scattering_bias(g: float) -> None:
    """Forward-scattering (g>0) mean cos(theta) must be close to g."""
    rng = np.random.default_rng(0)
    pf = HenyeyGreensteinPhaseFunction(g=g)
    theta = pf.sample(100_000, rng)
    mean_cos = float(np.mean(np.cos(theta)))
    # Analytical mean of cos(theta) for HG = g
    assert abs(mean_cos - g) < 0.01, (
        f"Mean cos(θ)={mean_cos:.4f} deviates from g={g:.4f}"
    )


def test_hg_invalid_g() -> None:
    """g outside (-1, 1) must raise ValueError."""
    with pytest.raises(ValueError):
        HenyeyGreensteinPhaseFunction(g=1.0)
    with pytest.raises(ValueError):
        HenyeyGreensteinPhaseFunction(g=-1.0)


def test_hg_cdf_build_and_sample() -> None:
    """CDF build + sample_from_cdf must match direct analytical sampling."""
    rng = np.random.default_rng(1)
    g = 0.7
    pf = HenyeyGreensteinPhaseFunction(g=g)

    # Build tabulated CDF
    theta_grid = np.linspace(0, np.pi, 2000)
    vsf = (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * np.cos(theta_grid)) ** 1.5)
    cdf, angles = pf.build_cdf(theta_grid, vsf)

    n = 50_000
    theta_tab = pf.sample_from_cdf(cdf, angles, n, rng)
    theta_ana = pf.sample(n, rng)

    # Both should have similar mean
    assert abs(np.mean(np.cos(theta_tab)) - g) < 0.02
    assert abs(np.mean(np.cos(theta_ana)) - g) < 0.02
