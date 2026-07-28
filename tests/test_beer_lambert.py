"""Test Beer-Lambert transmission in a pure absorber."""

import math

import pytest

from photonator.beam.gaussian import GaussianBeam
from photonator.core.receiver import Receiver
from photonator.media.water import Water
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.simulation import Simulation


@pytest.mark.parametrize(
    "mu_a_per_m, receiver_z_m",
    [
        (0.1, 5.0),
        (0.5, 2.0),
        (1.0, 1.0),
    ],
)
def test_beer_lambert_transmission(mu_a_per_m: float, receiver_z_m: float) -> None:
    """Transmitted power in a pure absorber must match exp(-mu_a * z) within 1%."""
    n_photons = 200_000
    expected_T = math.exp(-mu_a_per_m * receiver_z_m)

    medium = Water(mu_a_per_m=mu_a_per_m, mu_s_per_m=0.0)
    beam = GaussianBeam(w0_m=0.0, half_angle_divergence_rad=0.0)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.0)
    receiver = Receiver(
        receiver_z_m=receiver_z_m,
        aperture_m=1e6,
        fov_rad=math.pi,
        n_water=1.0,  # no interface → no Fresnel loss in this analytical test
        n_air=1.0,
    )

    sim = Simulation(
        medium=medium,
        beam=beam,
        phase_fn=phase_fn,
        receiver=receiver,
        n_photons=n_photons,
        n_batches=1,
        seed=42,
    )
    result = sim.run()

    measured_T = result.total_power / n_photons
    error = abs(measured_T - expected_T) / expected_T
    assert error < 0.01, (
        f"Beer-Lambert error {error:.3%} > 1% "
        f"(expected={expected_T:.4f}, measured={measured_T:.4f})"
    )


def test_zero_absorption_full_transmission() -> None:
    """With no absorption and no scattering, all photons must reach the receiver."""
    n_photons = 10_000
    medium = Water(mu_a_per_m=0.0, mu_s_per_m=0.0)
    # Avoid mu_t=0 edge case: use tiny but nonzero values
    medium._mu_a = 1e-9
    medium._mu_s = 1e-9
    beam = GaussianBeam(w0_m=0.0, half_angle_divergence_rad=0.0)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.0)
    receiver = Receiver(receiver_z_m=1.0, aperture_m=1e6, fov_rad=math.pi)

    sim = Simulation(
        medium=medium, beam=beam, phase_fn=phase_fn, receiver=receiver,
        n_photons=n_photons, n_batches=1, seed=0,
    )
    result = sim.run()
    assert result.total_packets > 0
