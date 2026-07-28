"""Tests for the CPU propagation loop and direction-cosine update."""

import math

import numpy as np

from photonator.beam.gaussian import GaussianBeam
from photonator.core.receiver import Receiver
from photonator.core.scattering import update_direction
from photonator.media.water import Water
from photonator.phase_functions.henyey_greenstein import HenyeyGreensteinPhaseFunction
from photonator.simulation import Simulation

# --- Direction cosine update tests ---

def test_update_direction_unit_vector() -> None:
    """Updated direction cosines must remain on the unit sphere."""
    rng = np.random.default_rng(0)
    n = 10_000
    ux = rng.uniform(-1, 1, n)
    uy = rng.uniform(-1, 1, n)
    uz = np.sqrt(np.maximum(1.0 - ux**2 - uy**2, 0.0))
    # Re-normalise
    norm = np.sqrt(ux**2 + uy**2 + uz**2)
    ux /= norm
    uy /= norm
    uz /= norm

    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)

    ux_new, uy_new, uz_new = update_direction(ux, uy, uz, theta, phi)
    norms = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_update_direction_near_axis() -> None:
    """Near-axis (uz≈1) case must also produce unit vectors."""
    n = 100
    ux = np.zeros(n)
    uy = np.zeros(n)
    uz = np.ones(n)  # exactly on axis
    theta = np.full(n, 0.1)
    phi = np.linspace(0, 2 * np.pi, n)

    ux_new, uy_new, uz_new = update_direction(ux, uy, uz, theta, phi)
    norms = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_update_direction_zero_scattering() -> None:
    """theta=0 → direction must remain unchanged."""
    n = 100
    rng = np.random.default_rng(1)
    ux = rng.uniform(-0.5, 0.5, n)
    uy = rng.uniform(-0.5, 0.5, n)
    uz = np.sqrt(np.maximum(1 - ux**2 - uy**2, 0.0))
    norm = np.sqrt(ux**2 + uy**2 + uz**2)
    ux /= norm
    uy /= norm
    uz /= norm

    theta = np.zeros(n)
    phi = rng.uniform(0, 2 * np.pi, n)

    ux_new, uy_new, uz_new = update_direction(ux, uy, uz, theta, phi)
    np.testing.assert_allclose(ux_new, ux, atol=1e-10)
    np.testing.assert_allclose(uy_new, uy, atol=1e-10)
    np.testing.assert_allclose(uz_new, uz, atol=1e-10)


# --- Propagation integration tests ---

def test_all_photons_terminate() -> None:
    """All photons must end as either detected or terminated (none remain active)."""
    medium = Water(mu_a_per_m=0.1, mu_s_per_m=0.5)
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.9)
    receiver = Receiver(receiver_z_m=5.0, aperture_m=1.0, fov_rad=math.pi / 2)

    sim = Simulation(
        medium=medium, beam=beam, phase_fn=phase_fn, receiver=receiver,
        n_photons=1_000, n_batches=1, seed=0,
    )
    result = sim.run()
    # total_packets is detected; reflected + (n - total_plane_crossings) are terminated
    assert result.n_photons == 1_000


def test_more_photons_more_power() -> None:
    """Doubling photons should roughly double received power."""
    medium = Water(mu_a_per_m=0.1, mu_s_per_m=0.5)
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.001)
    phase_fn = HenyeyGreensteinPhaseFunction(g=0.9)

    def run_n(n: int, seed: int) -> float:
        rx = Receiver(receiver_z_m=5.0, aperture_m=1.0, fov_rad=math.pi / 2)
        sim = Simulation(medium=medium, beam=beam, phase_fn=phase_fn, receiver=rx,
                         n_photons=n, n_batches=1, seed=seed)
        return sim.run().total_power

    p1 = run_n(5_000, 0)
    p2 = run_n(10_000, 0)
    # p2/p1 should be ≈ 2 within 20%
    assert 1.5 < p2 / max(p1, 1e-15) < 2.5


def test_gaussian_beam_positions_on_axis() -> None:
    """Collimated pencil beam (w0=0) must place all photons at (0,0)."""
    rng = np.random.default_rng(7)
    beam = GaussianBeam(w0_m=0.0, half_angle_divergence_rad=0.0)
    batch = beam.initialize(100, rng)
    # With w0=0, radius=0, so x=y=0
    np.testing.assert_allclose(batch.x_m, 0.0, atol=1e-14)
    np.testing.assert_allclose(batch.y_m, 0.0, atol=1e-14)
    np.testing.assert_allclose(batch.uz, 1.0, atol=1e-14)


# --- Per-photon direction continuity (regression for index-remap bug) ---

class _FlipEveryTenthPhaseFunction(HenyeyGreensteinPhaseFunction):
    """theta = pi for every 10th sample, 0 otherwise.

    Flipped photons reverse direction exactly (theta=pi is a pure negation
    of the direction vector) and drift backward until the z<0 filter
    terminates them MID-LOOP while other photons continue -- exactly the
    condition under which the old index-remapping bug paired surviving
    photons with other photons' previous directions.
    """

    def __init__(self) -> None:
        super().__init__(g=0.0)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        theta = np.zeros(n, dtype=np.float64)
        theta[::10] = np.pi
        return theta


def test_direction_continuity_under_terminations() -> None:
    """Each detected photon must arrive with ITS OWN initial direction.

    Under theta in {0, pi} scattering every photon's direction is always
    +/- its initial vector, and crossing the receiver plane (uz > 0)
    requires the + sign -- so detected direction == initial direction,
    photon by photon.  Mid-loop z<0 and roulette terminations shift the
    filter frames every iteration; the old bug then rotated survivors
    from other photons' directions and broke this equality.
    """
    from photonator.core.propagation import propagate_cpu

    n = 5_000
    rng = np.random.default_rng(123)
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.01)
    batch = beam.initialize(n, rng)
    ux0 = batch.ux.copy()
    uy0 = batch.uy.copy()
    uz0 = batch.uz.copy()

    medium = Water(mu_a_per_m=0.5, mu_s_per_m=0.5)
    receiver = Receiver(receiver_z_m=10.0, aperture_m=1e6, fov_rad=math.pi)

    _, rec_loc, _, _, n_packets = propagate_cpu(
        batch, medium, _FlipEveryTenthPhaseFunction(), receiver
    )

    # The scenario must exercise both outcomes: detections AND mid-loop
    # terminations (backward-drifting photons exiting z<0, plus roulette).
    assert n_packets > 0, "no photons detected -- scenario broken"
    assert n_packets < n, "no photons terminated -- filters never fired"

    det = batch.detected_mask
    np.testing.assert_allclose(rec_loc[:, 2], ux0[det], atol=1e-6)
    np.testing.assert_allclose(rec_loc[:, 3], uy0[det], atol=1e-6)
    np.testing.assert_allclose(rec_loc[:, 4], uz0[det], atol=1e-6)
