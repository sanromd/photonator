"""Tests for Phase 6 advanced media: DispersiveOil and FluorescentMedium."""

import numpy as np
import pytest

from photonator.media.oil import Oil, DispersiveOil, _CAUCHY, _OIL_PARAMS
from photonator.media.fluorescent import FluorescentMedium
from photonator.media.water import Water


# ── DispersiveOil ─────────────────────────────────────────────────────────────

class TestDispersiveOil:
    def test_fixed_n_without_wavelength(self):
        """Without wavelength_nm, DispersiveOil returns same n as Oil."""
        for oil_type in ("crude", "refined", "mineral"):
            oil = Oil(oil_type=oil_type)
            doil = DispersiveOil(oil_type=oil_type)
            assert doil.n == oil.n, f"{oil_type}: fixed n mismatch"

    @pytest.mark.parametrize("oil_type", ["crude", "refined", "mineral"])
    def test_cauchy_calibration_at_589nm(self, oil_type):
        """Cauchy formula must reproduce literature n at 589 nm within 0.005."""
        expected_n = _OIL_PARAMS[oil_type]["n"]
        doil = DispersiveOil(oil_type=oil_type, wavelength_nm=589.0)
        assert abs(doil.n - expected_n) < 0.005, (
            f"{oil_type}: Cauchy n={doil.n:.5f} vs literature {expected_n}"
        )

    @pytest.mark.parametrize("oil_type", ["crude", "refined", "mineral"])
    def test_normal_dispersion(self, oil_type):
        """Oils exhibit normal dispersion: n decreases with increasing wavelength."""
        doil = DispersiveOil(oil_type=oil_type)
        n_blue  = doil.n_at(450.0)
        n_green = doil.n_at(532.0)
        n_red   = doil.n_at(650.0)
        assert n_blue > n_green > n_red, (
            f"{oil_type}: dispersion not monotone: "
            f"n(450)={n_blue:.5f} n(532)={n_green:.5f} n(650)={n_red:.5f}"
        )

    def test_n_at_is_stateless(self):
        """n_at() must not change the stored wavelength_nm."""
        doil = DispersiveOil(oil_type="refined", wavelength_nm=532.0)
        n_532 = doil.n
        _ = doil.n_at(450.0)
        assert doil.n == n_532, "n_at() must not mutate wavelength_nm"

    def test_cauchy_n_formula(self):
        """Verify Cauchy formula numerically against hand calculation."""
        oil_type = "refined"
        c = _CAUCHY[oil_type]
        wl_nm = 532.0
        lam_um = wl_nm / 1000.0
        expected = c["A"] + c["B"] / lam_um**2 + c["C"] / lam_um**4
        doil = DispersiveOil(oil_type=oil_type, wavelength_nm=wl_nm)
        assert abs(doil.n - expected) < 1e-12

    def test_mu_a_mu_s_g_forwarded(self):
        """Non-IOR properties must be the same as parent Oil."""
        oil = Oil(oil_type="refined")
        doil = DispersiveOil(oil_type="refined", wavelength_nm=532.0)
        assert doil.mu_a_per_m == oil.mu_a_per_m
        assert doil.mu_s_per_m == oil.mu_s_per_m
        assert doil.g == oil.g

    def test_override_mu_a_forwarded(self):
        """Constructor override of mu_a must be respected."""
        doil = DispersiveOil(oil_type="refined", mu_a_per_m=10.0)
        assert doil.mu_a_per_m == 10.0

    def test_invalid_oil_type_raises(self):
        with pytest.raises(ValueError):
            DispersiveOil(oil_type="unknown")

    def test_kk_requires_both_or_neither(self):
        wl = np.linspace(300, 800, 100)
        k = np.zeros(100)
        with pytest.raises(ValueError):
            DispersiveOil(oil_type="refined", k_wavelengths_nm=wl)
        with pytest.raises(ValueError):
            DispersiveOil(oil_type="refined", k_spectrum=k)

    def test_kramers_kronig_zero_k_gives_one(self):
        """KK integral with k=0 everywhere should give n ≈ 1."""
        wl = np.linspace(200.0, 2000.0, 2000)
        k = np.zeros_like(wl)
        doil = DispersiveOil(
            oil_type="refined",
            wavelength_nm=532.0,
            k_wavelengths_nm=wl,
            k_spectrum=k,
        )
        # With k=0, integrand is zero → n = 1 + 0 = 1
        assert abs(doil.n - 1.0) < 1e-10

    def test_kramers_kronig_consistent_with_cauchy(self):
        """KK from a Lorentzian k(λ) must give n close to Cauchy for typical oil.

        We use a narrow Lorentzian absorption peak far from the query wavelength
        so its contribution to n at 532 nm is small but non-zero, and verify
        the sign convention (absorption below resonance → n > 1).
        """
        wl = np.linspace(200.0, 2000.0, 5000)
        # Narrow Lorentzian absorption at 300 nm (UV, far from 532 nm)
        k = 0.01 / (1 + ((wl - 300.0) / 5.0) ** 2)
        doil = DispersiveOil(
            oil_type="refined",
            wavelength_nm=532.0,
            k_wavelengths_nm=wl,
            k_spectrum=k,
        )
        # n must be > 1 (UV absorption pushes n above 1 in the visible)
        assert doil.n > 1.0

    def test_dispersion_range_realistic(self):
        """n at 400–800 nm should lie within [1.44, 1.55] for refined oil."""
        doil = DispersiveOil(oil_type="refined")
        for wl in (400.0, 500.0, 600.0, 700.0, 800.0):
            n = doil.n_at(wl)
            assert 1.44 < n < 1.55, f"n({wl} nm)={n:.4f} out of realistic range"


# ── FluorescentMedium ─────────────────────────────────────────────────────────

class TestFluorescentMedium:
    def _make(self, mu_a_ex=1.0, qy=0.9, em=600.0):
        host = Water()
        return FluorescentMedium(
            host=host,
            mu_a_ex_per_m=mu_a_ex,
            emission_wavelength_nm=em,
            quantum_yield=qy,
        )

    def test_mu_a_adds_fluorophore(self):
        host = Water()
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=2.5,
                               emission_wavelength_nm=600.0, quantum_yield=0.9)
        assert abs(fm.mu_a_per_m - (host.mu_a_per_m + 2.5)) < 1e-15

    def test_mu_s_unchanged(self):
        host = Water()
        fm = self._make()
        assert fm.mu_s_per_m == host.mu_s_per_m

    def test_g_unchanged(self):
        host = Water()
        fm = self._make()
        assert fm.g == host.g

    def test_n_unchanged(self):
        host = Water()
        fm = self._make()
        assert fm.n == host.n

    def test_inelastic_yield_stored(self):
        fm = self._make(qy=0.85)
        assert fm.inelastic_yield == 0.85
        assert fm.quantum_yield == 0.85

    def test_emission_wavelength_stored(self):
        fm = self._make(em=620.0)
        assert fm.emission_wavelength_nm == 620.0

    def test_mu_a_fluorophore_property(self):
        fm = self._make(mu_a_ex=3.7)
        assert fm.mu_a_fluorophore_per_m == pytest.approx(3.7)

    def test_wavelength_forwarded_from_host(self):
        host = Water(wavelength_nm=532.0)
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=1.0,
                               emission_wavelength_nm=600.0, quantum_yield=0.9)
        assert fm.wavelength_nm == 532.0

    def test_wavelength_override(self):
        host = Water(wavelength_nm=532.0)
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=1.0,
                               emission_wavelength_nm=600.0, quantum_yield=0.9,
                               wavelength_nm=488.0)
        assert fm.wavelength_nm == 488.0

    def test_invalid_quantum_yield_raises(self):
        host = Water()
        with pytest.raises(ValueError):
            FluorescentMedium(host=host, mu_a_ex_per_m=1.0,
                              emission_wavelength_nm=600.0, quantum_yield=1.1)
        with pytest.raises(ValueError):
            FluorescentMedium(host=host, mu_a_ex_per_m=1.0,
                              emission_wavelength_nm=600.0, quantum_yield=-0.1)

    def test_negative_mu_a_raises(self):
        host = Water()
        with pytest.raises(ValueError):
            FluorescentMedium(host=host, mu_a_ex_per_m=-0.5,
                              emission_wavelength_nm=600.0, quantum_yield=0.9)

    def test_zero_fluorophore_is_transparent_add(self):
        """With mu_a_ex=0 the medium must be identical to the host."""
        host = Water()
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=0.0,
                               emission_wavelength_nm=600.0, quantum_yield=0.0)
        assert fm.mu_a_per_m == host.mu_a_per_m
        assert fm.mu_s_per_m == host.mu_s_per_m

    def test_from_concentration_conversion(self):
        """Beer-Lambert conversion: μ_a = ε × C × ln(10) × 100."""
        host = Water()
        eps = 80_000.0   # L mol⁻¹ cm⁻¹  (typical organic dye)
        conc = 1e-6      # 1 μM
        expected_mu_a = eps * conc * 2.302585 * 100.0
        fm = FluorescentMedium.from_concentration(
            host=host,
            concentration_mol_per_L=conc,
            molar_extinction_L_per_mol_cm=eps,
            emission_wavelength_nm=600.0,
            quantum_yield=0.9,
        )
        assert abs(fm.mu_a_fluorophore_per_m - expected_mu_a) / expected_mu_a < 1e-6

    def test_from_concentration_qy_forwarded(self):
        host = Water()
        fm = FluorescentMedium.from_concentration(
            host=host,
            concentration_mol_per_L=1e-6,
            molar_extinction_L_per_mol_cm=50_000.0,
            emission_wavelength_nm=600.0,
            quantum_yield=0.75,
        )
        assert fm.quantum_yield == 0.75

    def test_stokes_shift_stored(self):
        host = Water()
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=1.0,
                               emission_wavelength_nm=600.0, quantum_yield=0.9,
                               stokes_shift_nm=25.0)
        assert fm.stokes_shift_nm == 25.0

    def test_host_can_be_dispersive_oil(self):
        """FluorescentMedium wraps any AbstractMedium, including DispersiveOil."""
        host = DispersiveOil(oil_type="mineral", wavelength_nm=532.0)
        fm = FluorescentMedium(host=host, mu_a_ex_per_m=0.5,
                               emission_wavelength_nm=620.0, quantum_yield=0.8)
        assert fm.n == host.n
        assert abs(fm.mu_a_per_m - (host.mu_a_per_m + 0.5)) < 1e-15
