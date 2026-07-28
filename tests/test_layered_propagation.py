"""Tests for propagation through axially layered media."""

import math

import numpy as np

from photonator.beam.gaussian import GaussianBeam
from photonator.core.photon import PhotonBatch
from photonator.core.propagation_layered import propagate_layered_cpu
from photonator.core.receiver import Receiver
from photonator.media.layers import Layer, LayeredMedium
from photonator.media.water import Water
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.simulation import Simulation


def _open_receiver(z_m: float) -> Receiver:
    """Receiver with no interface Fresnel, full FOV, huge aperture."""
    return Receiver(
        receiver_z_m=z_m, aperture_m=1e6, fov_rad=math.pi, n_water=1.0, n_air=1.0
    )


def test_two_layer_beer_lambert() -> None:
    """Pure absorbers with matched n: T = exp(-mu1*L1 - mu2*L2) within 1%.

    With identical refractive indices the interface Fresnel reflectance is
    exactly zero, so the layered loop must reproduce the analytic
    two-segment Beer-Lambert transmission.
    """
    mu1, mu2 = 0.3, 0.8
    z_interface, z_rec = 2.0, 4.0
    n_photons = 200_000

    layers = [
        Layer(0.0, z_interface, Water(mu_a_per_m=mu1, mu_s_per_m=0.0)),
        Layer(z_interface, 10.0, Water(mu_a_per_m=mu2, mu_s_per_m=0.0)),
    ]
    sim = Simulation(
        medium=LayeredMedium(layers),
        beam=GaussianBeam(w0_m=0.0, half_angle_divergence_rad=0.0),
        phase_fn=HenyeyGreensteinPhaseFunction(g=0.0),
        receiver=_open_receiver(z_rec),
        n_photons=n_photons,
        seed=11,
    )
    result = sim.run()

    expected_T = math.exp(-mu1 * z_interface - mu2 * (z_rec - z_interface))
    measured_T = result.total_power / n_photons
    error = abs(measured_T - expected_T) / expected_T
    assert error < 0.01, f"error {error:.3%} (expected {expected_T:.4f}, got {measured_T:.4f})"


def test_matched_layers_equal_homogeneous() -> None:
    """Two identical scattering layers must match a homogeneous run statistically."""
    water_kwargs = dict(mu_a_per_m=0.05, mu_s_per_m=0.4)
    n_photons = 30_000
    z_rec = 5.0

    def run(medium) -> float:
        sim = Simulation(
            medium=medium,
            beam=GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001),
            phase_fn=HenyeyGreensteinPhaseFunction(g=0.9),
            receiver=_open_receiver(z_rec),
            n_photons=n_photons,
            seed=13,
        )
        return sim.run().total_power

    p_homo = run(Water(**water_kwargs))
    p_layered = run(
        LayeredMedium(
            [
                Layer(0.0, 2.5, Water(**water_kwargs)),
                Layer(2.5, 10.0, Water(**water_kwargs)),
            ]
        )
    )
    rel_diff = abs(p_layered - p_homo) / p_homo
    assert rel_diff < 0.10, f"layered vs homogeneous power differ by {rel_diff:.1%}"


def test_snell_refraction_at_interface() -> None:
    """Photons crossing n1→n2 must exit with the Snell-refracted angle.

    Negligible attenuation makes the interface the only event; detected
    photons must all carry uz = cos(theta2) with sin(theta2) = n1 sin(theta1)/n2
    to within 1e-9 (Fresnel-reflected photons head backward and never arrive).
    """
    n1, n2 = 1.33, 1.46
    theta1 = math.radians(20.0)
    vacuum_ish = dict(mu_a_per_m=1e-9, mu_s_per_m=1e-9)

    layers = [
        Layer(0.0, 2.0, Water(**vacuum_ish, n=n1)),
        Layer(2.0, 10.0, Water(**vacuum_ish, n=n2)),
    ]
    batch = PhotonBatch(200, rng=np.random.default_rng(5))
    batch.ux[:] = math.sin(theta1)
    batch.uz[:] = math.cos(theta1)

    _, rec_loc, _, _, n_packets = propagate_layered_cpu(
        batch,
        LayeredMedium(layers),
        HenyeyGreensteinPhaseFunction(g=0.0),
        _open_receiver(6.0),
    )

    sin_theta2 = n1 * math.sin(theta1) / n2
    cos_theta2 = math.sqrt(1.0 - sin_theta2**2)
    assert n_packets > 150, "most photons should transmit at this angle"
    np.testing.assert_allclose(rec_loc[:, 4], cos_theta2, atol=1e-9)
    np.testing.assert_allclose(rec_loc[:, 2], sin_theta2, atol=1e-9)


def test_total_internal_reflection() -> None:
    """Beyond the critical angle every photon reflects and none is detected."""
    n1, n2 = 1.50, 1.33
    # Critical angle: arcsin(1.33/1.50) ~ 62.5 deg; launch at 70 deg
    theta1 = math.radians(70.0)
    vacuum_ish = dict(mu_a_per_m=1e-9, mu_s_per_m=1e-9)

    layers = [
        Layer(0.0, 2.0, Water(**vacuum_ish, n=n1)),
        Layer(2.0, 10.0, Water(**vacuum_ish, n=n2)),
    ]
    batch = PhotonBatch(100, rng=np.random.default_rng(6))
    batch.ux[:] = math.sin(theta1)
    batch.uz[:] = math.cos(theta1)

    _, _, _, _, n_packets = propagate_layered_cpu(
        batch,
        LayeredMedium(layers),
        HenyeyGreensteinPhaseFunction(g=0.0),
        _open_receiver(6.0),
    )
    assert n_packets == 0, "TIR photons must never reach the receiver"
    # All photons reflected downward and exited through z=0
    assert batch.n_active == 0


def test_per_layer_phase_function() -> None:
    """A layer-specific phase function must be used inside that layer."""

    class CountingPhaseFunction(HenyeyGreensteinPhaseFunction):
        def __init__(self) -> None:
            super().__init__(g=0.0)
            self.calls = 0

        def sample(self, n, rng):
            self.calls += 1
            return super().sample(n, rng)

    layer_pf = CountingPhaseFunction()
    global_pf = CountingPhaseFunction()
    layers = [
        Layer(0.0, 2.0, Water(mu_a_per_m=0.1, mu_s_per_m=0.5), phase_fn=layer_pf),
        Layer(2.0, 10.0, Water(mu_a_per_m=0.1, mu_s_per_m=0.5)),
    ]
    sim = Simulation(
        medium=LayeredMedium(layers),
        beam=GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001),
        phase_fn=global_pf,
        receiver=_open_receiver(5.0),
        n_photons=2_000,
        seed=7,
    )
    sim.run()
    assert layer_pf.calls > 0, "layer-specific phase function never sampled"
    assert global_pf.calls > 0, "global fallback never sampled in the second layer"
