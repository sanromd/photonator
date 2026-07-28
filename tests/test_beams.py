"""Unit tests for LaguerreGaussianBeam, BesselBeam, and AiryBeam."""

import numpy as np
import pytest

from photonator.beam.airy import AiryBeam
from photonator.beam.bessel import BesselBeam
from photonator.beam.gaussian import GaussianBeam
from photonator.beam.laguerre import LaguerreGaussianBeam

RNG = np.random.default_rng(42)
N = 50_000


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit_check(batch) -> float:
    """Return max deviation of |u| from 1.0."""
    mag = np.sqrt(batch.ux**2 + batch.uy**2 + batch.uz**2)
    return float(np.max(np.abs(mag - 1.0)))


# ── GaussianBeam (baseline sanity) ───────────────────────────────────────────

def test_gaussian_unit_cosines() -> None:
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.00075)
    batch = beam.initialize(N, np.random.default_rng(0))
    assert _unit_check(batch) < 1e-12


def test_gaussian_positions_rayleigh() -> None:
    """Gaussian radii should follow Rayleigh distribution: mean r = w0 * sqrt(pi/2) / sqrt(2)."""
    w0 = 0.001
    beam = GaussianBeam(w0_m=w0, half_angle_divergence_rad=0.0)
    batch = beam.initialize(N, np.random.default_rng(1))
    r = np.sqrt(batch.x_m**2 + batch.y_m**2)
    expected_mean = w0 * np.sqrt(np.pi / 2) / np.sqrt(2)
    assert abs(r.mean() - expected_mean) / expected_mean < 0.02


# ── LaguerreGaussianBeam ─────────────────────────────────────────────────────

@pytest.mark.parametrize("p,l", [(0, 1), (1, 0), (1, 2), (0, 3)])
def test_lg_unit_cosines(p: int, l: int) -> None:
    beam = LaguerreGaussianBeam(p=p, l=l, w0_m=0.001)
    batch = beam.initialize(N, np.random.default_rng(0))
    assert _unit_check(batch) < 1e-12, f"LG({p},{l}) direction cosines not unit"


@pytest.mark.parametrize("p,l", [(0, 1), (1, 0), (1, 2)])
def test_lg_positions_within_support(p: int, l: int) -> None:
    """All positions must lie within the envelope radius r_max."""
    w0 = 0.001
    beam = LaguerreGaussianBeam(p=p, l=l, w0_m=w0)
    r_max = w0 * np.sqrt(2.0 * (2 * p + abs(l) + 1))
    batch = beam.initialize(N, np.random.default_rng(1))
    r = np.sqrt(batch.x_m**2 + batch.y_m**2)
    assert float(r.max()) <= r_max * 1.001, (
        f"LG({p},{l}) photons outside envelope: r_max={r_max:.4f}, got {r.max():.4f}"
    )


def test_lg_donut_zero_on_axis() -> None:
    """LG_0^1 (donut mode) must have near-zero intensity on-axis.

    With |l|=1 the LG profile goes as r^2 near r=0, so the density of
    photons within a small central disk should be very low compared to the
    annular ring.
    """
    beam = LaguerreGaussianBeam(p=0, l=1, w0_m=0.001)
    batch = beam.initialize(100_000, np.random.default_rng(2))
    r = np.sqrt(batch.x_m**2 + batch.y_m**2)
    w0 = beam.w0_m

    # Fraction inside r < 0.1*w0  vs  fraction in ring 0.5*w0 < r < 1.5*w0
    frac_center = float(np.mean(r < 0.1 * w0))
    frac_ring = float(np.mean((r > 0.5 * w0) & (r < 1.5 * w0)))
    assert frac_center < 0.002, f"Too many photons on-axis: {frac_center:.4f}"
    assert frac_ring > 0.3, f"Too few photons in donut ring: {frac_ring:.4f}"


def test_lg_collimated_all_uz_one() -> None:
    """With zero divergence all photons travel straight along z."""
    beam = LaguerreGaussianBeam(p=0, l=1, w0_m=0.001, half_angle_divergence_rad=0.0)
    batch = beam.initialize(N, np.random.default_rng(3))
    assert np.allclose(batch.uz, 1.0, atol=1e-12)
    assert np.allclose(batch.ux, 0.0, atol=1e-12)
    assert np.allclose(batch.uy, 0.0, atol=1e-12)


def test_lg_higher_p_broader_profile() -> None:
    """Higher radial index p produces a broader transverse profile."""
    rng0 = np.random.default_rng(4)
    rng1 = np.random.default_rng(5)
    b0 = LaguerreGaussianBeam(p=0, l=1, w0_m=0.001).initialize(N, rng0)
    b1 = LaguerreGaussianBeam(p=2, l=1, w0_m=0.001).initialize(N, rng1)
    r0 = np.sqrt(b0.x_m**2 + b0.y_m**2).mean()
    r1 = np.sqrt(b1.x_m**2 + b1.y_m**2).mean()
    assert r1 > r0, f"Higher p should give larger mean radius: p=0 {r0:.4f}, p=2 {r1:.4f}"


# ── BesselBeam ────────────────────────────────────────────────────────────────

def test_bessel_unit_cosines() -> None:
    beam = BesselBeam(k_r_per_m=2000.0, theta_cone_rad=0.001, aperture_radius_m=0.01)
    batch = beam.initialize(N, np.random.default_rng(0))
    assert _unit_check(batch) < 1e-12


def test_bessel_positions_within_aperture() -> None:
    aperture = 0.01
    beam = BesselBeam(k_r_per_m=2000.0, theta_cone_rad=0.001, aperture_radius_m=aperture)
    batch = beam.initialize(N, np.random.default_rng(1))
    r = np.sqrt(batch.x_m**2 + batch.y_m**2)
    assert float(r.max()) <= aperture + 1e-10, (
        f"Bessel photon outside aperture: r_max={r.max():.6f} > aperture={aperture}"
    )


def test_bessel_uz_equals_cos_cone() -> None:
    """All uz values must equal cos(theta_cone_rad) exactly."""
    theta = 0.005
    beam = BesselBeam(k_r_per_m=2000.0, theta_cone_rad=theta, aperture_radius_m=0.01)
    batch = beam.initialize(N, np.random.default_rng(2))
    expected_uz = np.cos(theta)
    assert np.allclose(batch.uz, expected_uz, atol=1e-12)


def test_bessel_azimuthal_symmetry() -> None:
    """Bessel beam must be azimuthally symmetric: mean x and mean y near zero."""
    beam = BesselBeam(k_r_per_m=2000.0, theta_cone_rad=0.001, aperture_radius_m=0.01)
    batch = beam.initialize(N, np.random.default_rng(3))
    tol = 0.01 * beam.aperture_radius_m  # 1 % of aperture
    assert abs(batch.x_m.mean()) < tol
    assert abs(batch.y_m.mean()) < tol


def test_bessel_central_peak() -> None:
    """J0(0)=1, so the highest intensity is at the center (most photons near r=0)."""
    beam = BesselBeam(k_r_per_m=3000.0, theta_cone_rad=0.001, aperture_radius_m=0.005)
    batch = beam.initialize(100_000, np.random.default_rng(4))
    r = np.sqrt(batch.x_m**2 + batch.y_m**2)
    # The first ring of J0 is at k_r * r = 2.405  →  r ≈ 0.802 mm
    first_ring = 2.405 / beam.k_r_per_m
    frac_center = float(np.mean(r < first_ring / 2))
    frac_outer = float(np.mean(r > first_ring))
    assert frac_center > frac_outer, (
        f"Expected more photons in central lobe than outer rings: "
        f"center={frac_center:.3f}, outer={frac_outer:.3f}"
    )


# ── AiryBeam ─────────────────────────────────────────────────────────────────

def test_airy_unit_cosines() -> None:
    beam = AiryBeam(x0_m=0.001, y0_m=0.001)
    batch = beam.initialize(N, np.random.default_rng(0))
    assert _unit_check(batch) < 1e-12


def test_airy_collimated() -> None:
    """AiryBeam launches all photons along +z (uz=1, ux=uy=0)."""
    beam = AiryBeam(x0_m=0.001, y0_m=0.001)
    batch = beam.initialize(N, np.random.default_rng(1))
    assert np.allclose(batch.uz, 1.0, atol=1e-12)
    assert np.allclose(batch.ux, 0.0, atol=1e-12)
    assert np.allclose(batch.uy, 0.0, atol=1e-12)


def test_airy_positions_within_extent() -> None:
    """All photon positions must be within the sampling extent."""
    x0, y0 = 0.001, 0.001
    ext = 6.0
    beam = AiryBeam(x0_m=x0, y0_m=y0, extent_x0=ext, extent_y0=ext)
    batch = beam.initialize(N, np.random.default_rng(2))
    assert float(np.abs(batch.x_m).max()) <= x0 * ext + 1e-10
    assert float(np.abs(batch.y_m).max()) <= y0 * ext + 1e-10


def test_airy_transverse_scale() -> None:
    """Doubling x0_m should roughly double the RMS width in x."""
    rng0 = np.random.default_rng(6)
    rng1 = np.random.default_rng(7)
    b0 = AiryBeam(x0_m=0.001, y0_m=0.001).initialize(N, rng0)
    b1 = AiryBeam(x0_m=0.002, y0_m=0.001).initialize(N, rng1)
    rms0 = float(np.std(b0.x_m))
    rms1 = float(np.std(b1.x_m))
    ratio = rms1 / rms0
    assert 1.7 < ratio < 2.3, (
        f"Expected ~2× RMS ratio when x0 doubled, got {ratio:.3f}"
    )


def test_airy_asymmetric_scales() -> None:
    """With x0 ≠ y0 the profile must be wider in the larger-scale direction."""
    beam = AiryBeam(x0_m=0.002, y0_m=0.0005)
    batch = beam.initialize(N, np.random.default_rng(8))
    assert np.std(batch.x_m) > np.std(batch.y_m), (
        "Expected wider x extent when x0 > y0"
    )


def test_airy_first_lobe_denser_than_origin() -> None:
    """Airy function has its first local maximum at z ≈ −1.019 (not at z=0).

    The 1D marginal density of x is proportional to |Ai(x/x0)|^2, which peaks
    near x ≈ −1.019 * x0.  Therefore a strip around x = −x0 should contain
    more photons than an equal-width strip around x = 0.
    """
    x0 = 0.001
    beam = AiryBeam(x0_m=x0, y0_m=x0)
    batch = beam.initialize(100_000, np.random.default_rng(9))
    x = batch.x_m
    half = 0.3 * x0  # half-width of comparison strips
    n_origin = int(np.sum(np.abs(x) < half))
    n_first_lobe = int(np.sum(np.abs(x + x0) < half))  # strip centred at x = −x0
    assert n_first_lobe > n_origin, (
        f"First Airy lobe (n={n_first_lobe}) should exceed origin (n={n_origin})"
    )
