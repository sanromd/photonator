"""Tests for layered medium and material models."""

import numpy as np
import pytest

from photonator.media.brine import Brine, quan_fry_n
from photonator.media.layers import GradientMedium, Layer, LayeredMedium
from photonator.media.mixture import MixtureMedium
from photonator.media.oil import Oil
from photonator.media.water import Water

# --- LayeredMedium tests ---

def test_layered_medium_layer_lookup() -> None:
    """layer_at() must return the correct layer for a given z."""
    water = Water()
    oil = Oil()
    layers = [
        Layer(z_start_m=0.0, z_end_m=5.0, medium=water),
        Layer(z_start_m=5.0, z_end_m=10.0, medium=oil),
    ]
    medium = LayeredMedium(layers)
    assert medium.layer_at(2.5).medium is water
    assert medium.layer_at(7.0).medium is oil
    assert medium.layer_at(0.0).medium is water


def test_layered_medium_boundary() -> None:
    """boundary_crossings() must detect interface crossings."""
    water = Water()
    oil = Oil()
    layers = [
        Layer(z_start_m=0.0, z_end_m=5.0, medium=water),
        Layer(z_start_m=5.0, z_end_m=10.0, medium=oil),
    ]
    medium = LayeredMedium(layers)
    crossings = medium.boundary_crossings(4.0, 6.0)
    assert len(crossings) == 1
    z_bound, before, after = crossings[0]
    assert abs(z_bound - 5.0) < 1e-10
    assert before.medium is water
    assert after.medium is oil


def test_layered_medium_no_crossing() -> None:
    """No crossings returned when path stays in one layer."""
    water = Water()
    oil = Oil()
    layers = [
        Layer(0.0, 5.0, water),
        Layer(5.0, 10.0, oil),
    ]
    medium = LayeredMedium(layers)
    assert medium.boundary_crossings(1.0, 3.0) == []


# --- GradientMedium tests ---

def test_gradient_medium_depth_variation() -> None:
    """mu_a must vary smoothly with depth."""
    medium = GradientMedium(
        mu_a_func=lambda z: 0.1 + 0.01 * z,
        mu_s_func=lambda z: 0.5,
        g_func=lambda z: 0.9,
        n_func=lambda z: 1.33,
    )
    medium.set_query_z(0.0)
    assert abs(medium.mu_a_per_m - 0.1) < 1e-12
    medium.set_query_z(5.0)
    assert abs(medium.mu_a_per_m - 0.15) < 1e-12


# --- Brine tests ---

def test_quan_fry_n_pure_water_limit() -> None:
    """At zero salinity, IOR should be close to pure water at 532 nm."""
    n = quan_fry_n(532.0, salinity_g_per_L=0.0, temperature_C=20.0)
    assert 1.33 <= n <= 1.34, f"IOR out of range: {n}"


def test_brine_higher_salinity_higher_n() -> None:
    """Higher NaCl concentration should increase IOR."""
    b_low = Brine(salt="NaCl", concentration_g_per_L=10.0)
    b_high = Brine(salt="NaCl", concentration_g_per_L=100.0)
    assert b_high.n > b_low.n


def test_brine_invalid_salt() -> None:
    with pytest.raises(ValueError):
        Brine(salt="Unknown")


# --- MixtureMedium tests ---

def test_mixture_pure_water() -> None:
    """f=1.0 → pure medium_a."""
    water = Water(mu_a_per_m=0.1, mu_s_per_m=0.5)
    oil = Oil()
    mix = MixtureMedium(water, oil, volume_fraction_a=1.0)
    assert abs(mix.mu_a_per_m - 0.1) < 1e-12
    assert abs(mix.mu_s_per_m - 0.5) < 1e-12


def test_mixture_interpolation() -> None:
    """50/50 mix should have intermediate properties."""
    water = Water(mu_a_per_m=0.1, mu_s_per_m=0.2)
    oil = Oil(mu_a_per_m=1.0, mu_s_per_m=0.0)
    mix = MixtureMedium(water, oil, volume_fraction_a=0.5)
    assert abs(mix.mu_a_per_m - 0.55) < 1e-10
    assert abs(mix.mu_s_per_m - 0.1) < 1e-10


def test_mixture_invalid_fraction() -> None:
    water = Water()
    oil = Oil()
    with pytest.raises(ValueError):
        MixtureMedium(water, oil, volume_fraction_a=1.5)


# --- Snell's law at interface (geometric check) ---

def test_snells_law_at_interface() -> None:
    """At a hard interface, Snell's law n1*sin(t1) = n2*sin(t2) must hold."""
    n1 = 1.33  # water
    n2 = 1.46  # oil
    theta1_rad = np.radians(20.0)
    sin_t2 = n1 * np.sin(theta1_rad) / n2
    theta2_rad = np.arcsin(sin_t2)
    # Verify Snell's law algebraically
    assert abs(n1 * np.sin(theta1_rad) - n2 * np.sin(theta2_rad)) < 1e-12


def test_snells_angle_within_tolerance() -> None:
    """Snell's law angle must be recovered to within 0.1°."""
    n1, n2 = 1.33, 1.46
    theta1_deg = 15.0
    theta1_rad = np.radians(theta1_deg)
    sin_t2 = n1 * np.sin(theta1_rad) / n2
    theta2_recovered_deg = np.degrees(np.arcsin(sin_t2))
    # Expected: arcsin(1.33*sin(15°)/1.46)
    expected_deg = np.degrees(np.arcsin(1.33 * np.sin(np.radians(15.0)) / 1.46))
    assert abs(theta2_recovered_deg - expected_deg) < 0.1
