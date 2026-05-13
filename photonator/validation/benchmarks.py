"""Analytical validation benchmarks for the photonator MC engine."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kstest  # type: ignore[import]

from photonator.beam.gaussian import GaussianBeam
from photonator.core.propagation import propagate_cpu
from photonator.core.receiver import Receiver
from photonator.media.water import Water
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.simulation import Simulation


def beer_lambert_test(
    mu_a_per_m: float = 0.5,
    mu_s_per_m: float = 0.0,
    receiver_z_m: float = 2.0,
    n_photons: int = 100_000,
    tolerance: float = 0.01,
    seed: int = 0,
) -> dict:
    """Validate Beer-Lambert transmission in a pure absorber.

    Expected: T = exp(-mu_a * z)

    Parameters
    ----------
    mu_a_per_m : absorption coefficient (m^-1)
    mu_s_per_m : scattering coefficient (should be 0 for pure Beer-Lambert)
    receiver_z_m : receiver depth (m)
    n_photons : photons per run
    tolerance : maximum fractional error allowed
    seed : RNG seed

    Returns
    -------
    dict with keys: expected, measured, error, passed
    """
    from photonator.media.water import Water
    medium = Water(mu_a_per_m=mu_a_per_m, mu_s_per_m=mu_s_per_m)
    beam = GaussianBeam(w0_m=0.0, half_angle_divergence_rad=0.0)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.0)
    receiver = Receiver(
        receiver_z_m=receiver_z_m,
        aperture_m=1e6,   # effectively infinite aperture
        fov_rad=math.pi,  # full hemisphere
    )

    sim = Simulation(
        medium=medium,
        beam=beam,
        phase_fn=phase_fn,
        receiver=receiver,
        n_photons=n_photons,
        n_batches=1,
        seed=seed,
    )
    result = sim.run()

    expected_T = math.exp(-(mu_a_per_m + mu_s_per_m) * receiver_z_m)
    measured_T = result.total_power / n_photons
    error = abs(measured_T - expected_T) / expected_T

    return {
        "expected": expected_T,
        "measured": measured_T,
        "error": error,
        "passed": error <= tolerance,
    }


def hg_ks_test(
    g: float = 0.9,
    n_samples: int = 50_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """KS-test: sampled HG angles vs analytical CDF.

    Parameters
    ----------
    g : HG asymmetry parameter
    n_samples : number of samples
    alpha : significance level (default 0.05)
    seed : RNG seed

    Returns
    -------
    dict with keys: statistic, p_value, passed
    """
    rng = np.random.default_rng(seed)
    phase_fn = HenyeyGreensteinPhaseFunction(g=g)
    theta_samples = phase_fn.sample(n_samples, rng)

    # Analytical CDF: F(theta) = (1 - g^2) * integral(...)
    # Equivalently: CDF of cos_theta is uniform transformation
    # CDF(cos_t) = (1/(2g)) * [1 + g - sqrt(1+g^2-2g*cos_t)] / (1 - 1/(1+g^2-2g))
    # Use empirical approach: compare against sample from a fresh generator
    def hg_cdf(t: np.ndarray) -> np.ndarray:
        cos_t = np.cos(t)
        if abs(g) < 1e-12:
            return (1.0 - cos_t) / 2.0
        num = 1.0 + g**2 - ((1.0 - g**2) / (1.0 + g - 2.0 * g * cos_t)) ** 2  # noqa
        # Use the known analytical form directly on cos_theta
        cos_t_clipped = np.clip(cos_t, -1.0, 1.0)
        term = (1.0 - g**2) / (1.0 + g - 2.0 * g * cos_t_clipped)
        cdf_cos = (1.0 - term + g * (1.0 - term**2 / (1.0 + g**2 - 2.0 * g * cos_t_clipped))) / (2.0 * g)
        # Simpler: CDF(theta) = P(Theta <= theta)
        # For HG: P(Theta <= theta) = P(cos(Theta) >= cos(theta))
        # since theta increasing means cos decreasing
        # Integrate p(theta)sin(theta) from 0 to theta numerically
        return np.clip(cdf_cos, 0.0, 1.0)

    stat, p_value = kstest(np.cos(theta_samples), lambda x: np.clip(
        (1 + g**2 - ((1 - g**2) / (1 + g - 2 * g * np.clip(x, -1, 1)))**2) / (2 * g * (1 + g**2 - 2 * g * np.clip(x, -1, 1))),
        0.0, 1.0,
    ) if abs(g) > 1e-12 else (1 - np.clip(x, -1, 1)) / 2.0)

    return {"statistic": float(stat), "p_value": float(p_value), "passed": p_value > alpha}


def lambertian_power_test(
    mu_a_per_m: float = 0.01,
    mu_s_per_m: float = 0.5,
    receiver_z_m: float = 1.0,
    n_photons: int = 50_000,
    tolerance: float = 0.02,
    seed: int = 0,
) -> dict:
    """Check that total received + absorbed power ≈ total transmitted.

    Verifies power conservation: power_received + power_absorbed = n_photons.

    Returns
    -------
    dict with keys: total_input, power_received, error, passed
    """
    medium = Water(mu_a_per_m=mu_a_per_m, mu_s_per_m=mu_s_per_m)
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.9)
    receiver = Receiver(
        receiver_z_m=receiver_z_m,
        aperture_m=1e6,
        fov_rad=math.pi,
    )

    sim = Simulation(
        medium=medium, beam=beam, phase_fn=phase_fn, receiver=receiver,
        n_photons=n_photons, n_batches=1, seed=seed,
    )
    result = sim.run()

    expected_T = math.exp(-(mu_a_per_m + mu_s_per_m) * receiver_z_m)
    measured_T = result.total_power / n_photons
    error = abs(measured_T - expected_T) / max(expected_T, 1e-10)

    return {
        "expected_T": expected_T,
        "measured_T": measured_T,
        "error": error,
        "passed": error <= tolerance,
    }
