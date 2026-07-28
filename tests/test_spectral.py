"""Tests for Phase 4 spectral/wideband support."""

import math

import numpy as np
import pytest

from photonator.beam.gaussian import GaussianBeam
from photonator.core.receiver import Receiver
from photonator.media.oil import Oil
from photonator.media.water import Water, pope_fry_absorption_m_inv
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.spectral import SpectralSimulation


def _open_receiver(z_m: float) -> Receiver:
    return Receiver(
        receiver_z_m=z_m, aperture_m=1e6, fov_rad=math.pi, n_water=1.0, n_air=1.0
    )


# ── Water spectral model ─────────────────────────────────────────────────────

def test_pope_fry_table_shape() -> None:
    """Red water absorption must far exceed blue-green (well-known physics)."""
    assert pope_fry_absorption_m_inv(600.0) > 10 * pope_fry_absorption_m_inv(450.0)
    # Absorption minimum sits in the blue, near 420 nm
    assert pope_fry_absorption_m_inv(420.0) < pope_fry_absorption_m_inv(500.0)


def test_water_at_wavelength() -> None:
    w = Water(mu_s_per_m=0.05, g=0.9)
    w600 = w.at_wavelength(600.0)
    w450 = w.at_wavelength(450.0)
    assert w600.mu_a_per_m > w450.mu_a_per_m
    assert w600.wavelength_nm == 600.0
    # Non-absorption properties carry over
    assert w600.mu_s_per_m == w.mu_s_per_m
    assert w600.g == w.g


def test_default_at_wavelength_is_identity() -> None:
    """Media without spectral data return themselves unchanged."""
    oil = Oil(oil_type="mineral")
    assert oil.at_wavelength(600.0) is oil


# ── SpectralSimulation ───────────────────────────────────────────────────────

def _spectral(wavelengths, weights=None, n_photons=15_000, seed=0):
    return SpectralSimulation(
        medium=Water(),
        beam=GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001),
        phase_fn=HenyeyGreensteinPhaseFunction(g=0.93),
        receiver=_open_receiver(8.0),
        wavelengths_nm=wavelengths,
        spectral_weights=weights,
        n_photons=n_photons,
        seed=seed,
    )


def test_spectral_wavelength_dependence() -> None:
    """Blue-green must transmit far better than red through 8 m of water."""
    result = _spectral([450.0, 600.0]).run()
    assert result.power[0] > 2.0 * result.power[1], (
        f"expected strong blue advantage, got power={result.power}"
    )
    assert result.total_power == pytest.approx(float(np.sum(result.power)))
    assert len(result.bin_results) == 2


def test_spectral_weights_scale_bins() -> None:
    """Doubling one bin's source weight doubles its share of received power."""
    flat = _spectral([450.0, 550.0], weights=[1.0, 1.0], seed=5).run()
    tilted = _spectral([450.0, 550.0], weights=[2.0, 1.0], seed=5).run()
    # Same seed → same per-bin transmission; only the weighting differs:
    # flat gives 1/2 per bin, tilted gives 2/3 and 1/3.
    ratio = (tilted.power[0] / flat.power[0]) / (tilted.power[1] / flat.power[1])
    assert ratio == pytest.approx(2.0, rel=1e-9)


def test_spectral_validation_errors() -> None:
    with pytest.raises(ValueError):
        _spectral([])
    with pytest.raises(ValueError):
        _spectral([450.0, 550.0], weights=[1.0])
    with pytest.raises(ValueError):
        _spectral([450.0, 550.0], weights=[1.0, -1.0])
