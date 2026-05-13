"""Brine optical model using Quan & Fry (1995) extended Sellmeier equation.

Reference:
    Quan, X. & Fry, E. S. (1995). Empirical equation for the index of refraction
    of seawater. Applied Optics, 34(18), 3477–3480.

Absorption and scattering are approximated by scaling pure-water values
with an empirical salinity correction.
"""

from __future__ import annotations

import math

from photonator.media.base import AbstractMedium


# Quan & Fry (1995) coefficients for NaCl brine (seawater model)
_QF_N0 = 1.31405
_QF_N1 = 1.779e-4
_QF_N2 = -1.05e-6
_QF_N3 = 1.6e-8
_QF_N4 = -2.02e-6
_QF_N5 = 15.868
_QF_N6 = 0.01155
_QF_N7 = -0.00423
_QF_N8 = -4382.0
_QF_N9 = 1.1455e6

# Scattering concentration coefficients differ by salt species; NaCl default
_SALT_SCATTER_SCALE: dict[str, float] = {
    "NaCl": 1.0,
    "KCl": 0.95,
    "CaCl2": 1.12,
}

# Pure water baseline (Pope & Fry at 532 nm)
_PURE_WATER_MU_A_532: float = 0.0088   # m^-1
_PURE_WATER_MU_S_532: float = 0.037    # m^-1 (Petzold clear)
_PURE_WATER_G: float = 0.93


def quan_fry_n(
    wavelength_nm: float,
    salinity_g_per_L: float,
    temperature_C: float = 20.0,
) -> float:
    """Compute refractive index of NaCl brine via Quan & Fry (1995).

    Parameters
    ----------
    wavelength_nm : wavelength (nm)
    salinity_g_per_L : NaCl concentration (g/L ≈ g/kg for dilute solutions)
    temperature_C : temperature (°C)
    """
    lam = wavelength_nm
    S = salinity_g_per_L
    T = temperature_C
    n = (
        _QF_N0
        + (_QF_N1 + _QF_N2 * T + _QF_N3 * T**2) * S
        + _QF_N4 * T**2
        + (_QF_N5 + _QF_N6 * S + _QF_N7 * T) / lam
        + _QF_N8 / lam**2
        + _QF_N9 / lam**3
    )
    return float(n)


class Brine(AbstractMedium):
    """Optical model for saline water (NaCl, KCl, or CaCl₂).

    Parameters
    ----------
    salt : "NaCl", "KCl", or "CaCl2"
    concentration_g_per_L : salt concentration (g/L)
    wavelength_nm : operating wavelength (nm); used for IOR calculation
    temperature_C : temperature (°C)
    """

    def __init__(
        self,
        salt: str = "NaCl",
        concentration_g_per_L: float = 35.0,
        wavelength_nm: float = 532.0,
        temperature_C: float = 20.0,
    ) -> None:
        if salt not in _SALT_SCATTER_SCALE:
            raise ValueError(f"salt must be one of {list(_SALT_SCATTER_SCALE)}")
        self.salt = salt
        self.concentration_g_per_L = concentration_g_per_L
        self.wavelength_nm = wavelength_nm
        self.temperature_C = temperature_C

        scale = _SALT_SCATTER_SCALE[salt]
        # Salinity increases scattering slightly (empirical linear approximation)
        salinity_factor = 1.0 + 0.001 * concentration_g_per_L
        self._mu_s = _PURE_WATER_MU_S_532 * salinity_factor * scale
        # Absorption: salt adds negligible absorption at visible wavelengths
        self._mu_a = _PURE_WATER_MU_A_532

        if salt == "NaCl":
            self._n = quan_fry_n(wavelength_nm, concentration_g_per_L, temperature_C)
        else:
            # Fallback: linear IOR approximation for KCl/CaCl2
            dn_dc = {"KCl": 1.5e-4, "CaCl2": 2.0e-4}[salt]
            self._n = 1.333 + dn_dc * concentration_g_per_L

    @property
    def mu_a_per_m(self) -> float:
        return self._mu_a

    @property
    def mu_s_per_m(self) -> float:
        return self._mu_s

    @property
    def g(self) -> float:
        return _PURE_WATER_G

    @property
    def n(self) -> float:
        return self._n
