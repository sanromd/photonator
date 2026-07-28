"""Tests for Phase 3 inelastic (fluorescence) propagation."""

import math

from photonator.beam.gaussian import GaussianBeam
from photonator.core.receiver import Receiver
from photonator.media.fluorescent import FluorescentMedium
from photonator.media.water import Water
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.simulation import Simulation


def _open_receiver(z_m: float) -> Receiver:
    return Receiver(
        receiver_z_m=z_m, aperture_m=1e6, fov_rad=math.pi, n_water=1.0, n_air=1.0
    )


def _sim(medium, receiver_z_m: float, n_photons: int = 20_000, seed: int = 0) -> Simulation:
    return Simulation(
        medium=medium,
        beam=GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001),
        phase_fn=HenyeyGreensteinPhaseFunction(g=0.9),
        receiver=_open_receiver(receiver_z_m),
        n_photons=n_photons,
        seed=seed,
    )


def test_fluorescence_produces_emission_channel() -> None:
    """An active fluorophore must yield converted photons and detected emission."""
    host = Water(mu_a_per_m=0.01, mu_s_per_m=0.2)
    fm = FluorescentMedium(
        host, mu_a_ex_per_m=0.5, emission_wavelength_nm=600.0, quantum_yield=0.9
    )
    result = _sim(fm, receiver_z_m=3.0).run()

    assert result.fluoresced_photons > 0
    assert result.fluorescent_packets > 0
    assert result.fluorescent_power > 0.0
    assert result.emission_wavelength_nm == 600.0


def test_zero_quantum_yield_disables_fluorescence() -> None:
    """QY=0 means the fluorophore is a pure absorber: no emission channel."""
    host = Water(mu_a_per_m=0.01, mu_s_per_m=0.2)
    fm = FluorescentMedium(
        host, mu_a_ex_per_m=0.5, emission_wavelength_nm=600.0, quantum_yield=0.0
    )
    result = _sim(fm, receiver_z_m=3.0).run()

    assert result.fluoresced_photons == 0
    assert result.fluorescent_packets == 0
    assert result.fluorescent_power == 0.0
    assert result.emission_wavelength_nm is None


def test_dominant_fluorophore_converts_nearly_all() -> None:
    """With mu_a_f >> (mu_s, host mu_a) and QY=1, conversion prob per
    interaction is ~1, so the converted fraction must approach the
    interaction probability 1 - exp(-c * z_rec)."""
    host = Water(mu_a_per_m=1e-9, mu_s_per_m=1e-9)
    mu_a_f = 1.0
    z_rec = 2.0
    n = 40_000
    fm = FluorescentMedium(
        host, mu_a_ex_per_m=mu_a_f, emission_wavelength_nm=600.0, quantum_yield=1.0
    )
    result = _sim(fm, receiver_z_m=z_rec, n_photons=n, seed=3).run()

    expected_fraction = 1.0 - math.exp(-mu_a_f * z_rec)  # ~0.865
    measured_fraction = result.fluoresced_photons / n
    assert abs(measured_fraction - expected_fraction) < 0.02, (
        f"converted {measured_fraction:.3f}, expected {expected_fraction:.3f}"
    )
    # Ballistic survivors make up the elastic channel: ~exp(-c*z) of launch power
    ballistic = result.total_power / n
    assert abs(ballistic - math.exp(-mu_a_f * z_rec)) < 0.02


def test_energy_conservation_bound() -> None:
    """Elastic + fluorescent received power can never exceed launched power."""
    host = Water(mu_a_per_m=0.05, mu_s_per_m=0.5)
    fm = FluorescentMedium(
        host, mu_a_ex_per_m=0.3, emission_wavelength_nm=580.0, quantum_yield=1.0
    )
    n = 20_000
    result = _sim(fm, receiver_z_m=2.0, n_photons=n, seed=4).run()
    assert result.total_power + result.fluorescent_power < n


def test_emission_medium_drops_fluorophore() -> None:
    """The emission pass must propagate in the host, not the doped medium."""
    host = Water(mu_a_per_m=0.02, mu_s_per_m=0.3)
    fm = FluorescentMedium(
        host, mu_a_ex_per_m=2.0, emission_wavelength_nm=600.0, quantum_yield=0.5
    )
    em = fm.emission_medium()
    # Fluorophore excitation absorption (2.0) must be gone; the host's
    # spectral model replaces mu_a with the Pope & Fry value at 600 nm.
    assert em.mu_a_per_m < 1.0
    assert em.mu_s_per_m == host.mu_s_per_m
    assert em.wavelength_nm == 600.0


def test_plain_medium_has_no_fluorescence_channel() -> None:
    """A medium without hooks must leave the fluorescence fields zeroed."""
    result = _sim(Water(mu_a_per_m=0.05, mu_s_per_m=0.3), receiver_z_m=2.0).run()
    assert result.fluoresced_photons == 0
    assert result.fluorescent_power == 0.0
    assert result.emission_wavelength_nm is None
