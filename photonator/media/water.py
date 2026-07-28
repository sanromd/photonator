"""Pure water optical model.

Absorption: Pope & Fry (1997) pure-water spectrum, 400-700 nm.
Scattering: configurable; defaults to clear ocean (Petzold clear water).
IOR: n = 1.33 (seawater at visible wavelengths, matching MATLAB).
"""

from __future__ import annotations

import numpy as np

from photonator.constants import N_WATER
from photonator.media.base import AbstractMedium

# Pure-water absorption spectrum (m^-1), approximate values interpolated
# from Pope & Fry (1997), Table 3 ("Absorption spectrum of pure water.
# II. Integrating cavity measurements", Appl. Opt. 36, 8710).
# 10-nm grid, 400-700 nm.  Minimum is near 418 nm (~0.0044 m^-1).
_POPE_FRY_WL_NM = np.array(
    [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
     510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
     610, 620, 630, 640, 650, 660, 670, 680, 690, 700],
    dtype=np.float64,
)
_POPE_FRY_ABS_M_INV = np.array(
    [0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979, 0.0106,
     0.0127, 0.0150, 0.0204, 0.0325, 0.0409, 0.0434, 0.0474, 0.0565,
     0.0619, 0.0695, 0.0896, 0.1351, 0.2224, 0.2644, 0.2755, 0.2916,
     0.3108, 0.340, 0.410, 0.439, 0.465, 0.516, 0.624],
    dtype=np.float64,
)


def pope_fry_absorption_m_inv(wavelength_nm: float) -> float:
    """Interpolate the Pope & Fry (1997) pure-water absorption coefficient.

    Parameters
    ----------
    wavelength_nm : wavelength (nm); clamped to the 400-700 nm table range

    Returns
    -------
    absorption coefficient (m^-1)
    """
    return float(np.interp(wavelength_nm, _POPE_FRY_WL_NM, _POPE_FRY_ABS_M_INV))


# Pope & Fry absorption at the 532 nm default operating wavelength.
_POPE_FRY_532_NM_M_INV: float = pope_fry_absorption_m_inv(532.0)

# Petzold (1972) clear ocean: c = 0.151, a = 0.114  →  b = 0.037
_PETZOLD_CLEAR_C_M_INV: float = 0.151
_PETZOLD_CLEAR_A_M_INV: float = 0.114
_PETZOLD_CLEAR_B_M_INV: float = _PETZOLD_CLEAR_C_M_INV - _PETZOLD_CLEAR_A_M_INV
_PETZOLD_CLEAR_G: float = 0.93


class Water(AbstractMedium):
    """Optical model for water.

    Parameters
    ----------
    mu_a_per_m : absorption coefficient (m^-1); defaults to Pope & Fry at 532 nm
    mu_s_per_m : scattering coefficient (m^-1); defaults to Petzold clear ocean
    g : asymmetry parameter; defaults to 0.93
    n : refractive index; defaults to 1.33
    turbidity : multiplicative scattering scale factor (1.0 = clear ocean)
    wavelength_nm : operating wavelength (nm)
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

    def at_wavelength(self, wavelength_nm: float) -> Water:
        """Return a Water instance with absorption from the Pope & Fry table.

        Absorption is interpolated at ``wavelength_nm`` (any explicit
        ``mu_a_per_m`` override is replaced by the table value); scattering,
        asymmetry, and refractive index are carried over unchanged —
        spectral scattering is not modelled.
        """
        return Water(
            mu_a_per_m=pope_fry_absorption_m_inv(wavelength_nm),
            mu_s_per_m=self._mu_s,
            g=self._g,
            n=self._n,
            turbidity=1.0,
            wavelength_nm=wavelength_nm,
        )
