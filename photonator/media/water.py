"""Pure water optical model.

Absorption: Pope & Fry (1997) value at 532 nm.
Scattering: configurable; defaults to clear ocean (Petzold clear water).
IOR: n = 1.33 (seawater at visible wavelengths, matching MATLAB).
"""

from __future__ import annotations

from photonator.constants import N_WATER
from photonator.media.base import AbstractMedium

# Pope & Fry (1997) absorption at 532 nm: 0.0088 m^-1
# Segelstein (1981) value at 532 nm: ~0.0093 m^-1
# Using conservative Pope & Fry estimate as default
_POPE_FRY_532_NM_M_INV: float = 0.0088

# Petzold (1972) clear ocean: c = 0.151, a = 0.114  →  b = 0.037
_PETZOLD_CLEAR_C_M_INV: float = 0.151
_PETZOLD_CLEAR_A_M_INV: float = 0.114
_PETZOLD_CLEAR_B_M_INV: float = _PETZOLD_CLEAR_C_M_INV - _PETZOLD_CLEAR_A_M_INV
_PETZOLD_CLEAR_G: float = 0.93


class Water(AbstractMedium):
    """Optical model for water.

    Parameters
    ----------
    mu_a_per_m : absorption coefficient (m^-1); defaults to Pope & Fry 532 nm
    mu_s_per_m : scattering coefficient (m^-1); defaults to Petzold clear ocean
    g : asymmetry parameter; defaults to 0.93
    n : refractive index; defaults to 1.33
    turbidity : multiplicative scattering scale factor (1.0 = clear ocean)
    wavelength_nm : operating wavelength stub for spectral extension
    """

    def __init__(
        self,
        mu_a_per_m: float = _POPE_FRY_532_NM_M_INV,
        mu_s_per_m: float = _PETZOLD_CLEAR_B_M_INV,
        g: float = _PETZOLD_CLEAR_G,
        n: float = N_WATER,
        turbidity: float = 1.0,
        wavelength_nm: float | None = 532.0,
    ) -> None:
        self._mu_a = mu_a_per_m
        self._mu_s = mu_s_per_m * turbidity
        self._g = g
        self._n = n
        self.wavelength_nm = wavelength_nm

    @property
    def mu_a_per_m(self) -> float:
        return self._mu_a

    @property
    def mu_s_per_m(self) -> float:
        return self._mu_s

    @property
    def g(self) -> float:
        return self._g

    @property
    def n(self) -> float:
        return self._n
