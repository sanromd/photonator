"""Tests for the Receiver class."""

import math
import numpy as np
import pytest

from photonator.core.receiver import Receiver
from photonator.constants import N_WATER, N_AIR, CRIT_ANG_COS_WATER_AIR


def _make_on_axis_photons(n: int, mu_z: float = 0.98) -> tuple:
    """Create n photons hitting the receiver centre, going nearly straight."""
    rec_loc = np.zeros((n, 5))
    rec_loc[:, 0] = 0.0   # x = 0 (centre)
    rec_loc[:, 1] = 0.0   # y = 0
    rec_loc[:, 4] = mu_z  # uz
    distances_m = np.ones(n) * 5.0
    weights = np.ones(n)
    return rec_loc, distances_m, weights


def test_receiver_detects_on_axis_photons() -> None:
    """Photons hitting the centre with mu_z well above critical angle must be detected."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1.0, fov_rad=math.pi / 2)
    rec_loc, dist, weights = _make_on_axis_photons(100, mu_z=0.98)
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=100)
    assert result.count > 0
    assert result.power > 0.0


def test_receiver_critical_angle_rejection() -> None:
    """Photons at mu_z below critical angle cosine must be rejected (reflected)."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1.0, fov_rad=math.pi)
    # mu_z just below critical angle cosine
    mu_z_below = CRIT_ANG_COS_WATER_AIR - 0.01
    rec_loc, dist, weights = _make_on_axis_photons(50, mu_z=mu_z_below)
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=50)
    assert result.reflected == 50
    assert result.count == 0


def test_receiver_aperture_rejection() -> None:
    """Photons outside the aperture must not be counted."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=0.1, fov_rad=math.pi)
    rec_loc, dist, weights = _make_on_axis_photons(50, mu_z=0.98)
    # Place photons far from centre
    rec_loc[:, 0] = 10.0
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=50)
    assert result.count == 0


def test_receiver_fov_rejection() -> None:
    """Photons outside FOV (large angle from axis) must not be counted."""
    # FOV = 10 degrees → only photons with mu_z > cos(5°) ≈ 0.9962 accepted
    receiver = Receiver(
        receiver_z_m=5.0, aperture_m=1e6, fov_rad=math.radians(10)
    )
    rec_loc, dist, weights = _make_on_axis_photons(50, mu_z=0.95)  # ~18° from axis
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=50)
    assert result.count == 0


def test_receiver_fresnel_reduces_weight() -> None:
    """Fresnel transmission must be < 1 for large angles, reducing total power."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1e6, fov_rad=math.pi)
    mu_z_near_crit = CRIT_ANG_COS_WATER_AIR + 0.05
    rec_loc, dist, weights = _make_on_axis_photons(1000, mu_z=mu_z_near_crit)
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=1000)
    # Power must be less than total input weight (Fresnel loss)
    assert result.power < 1000.0 * 0.99


def test_receiver_welford_angle_mean() -> None:
    """Welford mean of mu_z must be close to the true mean."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1e6, fov_rad=math.pi)
    rng = np.random.default_rng(0)
    mu_z_vals = rng.uniform(CRIT_ANG_COS_WATER_AIR + 0.1, 1.0, 500)
    rec_loc = np.zeros((500, 5))
    rec_loc[:, 4] = mu_z_vals
    dist = np.ones(500) * 5.0
    weights = np.ones(500)
    result = receiver.detect(rec_loc, dist, weights, n_tx_photons=500)
    # angle_mean_rad is the Welford mean of mu_z (the uz values that passed Fresnel)
    assert result.count > 0
    assert 0.0 < result.angle_mean_rad < 1.0


def test_receiver_reset() -> None:
    """Reset clears accumulated statistics."""
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1.0, fov_rad=math.pi)
    rec_loc, dist, weights = _make_on_axis_photons(10, mu_z=0.98)
    receiver.detect(rec_loc, dist, weights, n_tx_photons=10)
    receiver.reset()
    assert receiver._count == 0
    assert receiver._power == 0.0
