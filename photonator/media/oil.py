"""Oil optical model (crude and refined petroleum products).

IOR values from:
  Millán-Merino et al. (2020), Energies, for crude oils ~1.46–1.52
  Refined mineral oil: ~1.46 at 589 nm

Absorption is highly variable; conservative estimates used as defaults.
"""

from __future__ import annotations

from photonator.media.base import AbstractMedium


# Literature IOR ranges for common oil types
_OIL_PARAMS: dict[str, dict[str, float]] = {
    "crude": {
        "n": 1.50,         # typical crude oil IOR
        "mu_a_per_m": 5.0, # approximate at 532 nm (highly variable)
        "mu_s_per_m": 0.1,
        "g": 0.85,
    },
    "refined": {
        "n": 1.46,
        "mu_a_per_m": 0.5,
        "mu_s_per_m": 0.05,
        "g": 0.85,
    },
    "mineral": {
        "n": 1.46,
        "mu_a_per_m": 0.2,
        "mu_s_per_m": 0.02,
        "g": 0.85,
    },
}


class Oil(AbstractMedium):
    """Optical model for petroleum oils.

    Parameters
    ----------
    oil_type : "crude", "refined", or "mineral"
    mu_a_per_m : override absorption coefficient (m^-1)
    mu_s_per_m : override scattering coefficient (m^-1)
    g : override asymmetry parameter
    n : override refractive index
    wavelength_nm : stub for spectral extension
    """

    def __init__(
        self,
        oil_type: str = "refined",
        mu_a_per_m: float | None = None,
        mu_s_per_m: float | None = None,
        g: float | None = None,
        n: float | None = None,
        wavelength_nm: float | None = None,
    ) -> None:
        if oil_type not in _OIL_PARAMS:
            raise ValueError(f"oil_type must be one of {list(_OIL_PARAMS)}")
        defaults = _OIL_PARAMS[oil_type]
        self._mu_a = mu_a_per_m if mu_a_per_m is not None else defaults["mu_a_per_m"]
        self._mu_s = mu_s_per_m if mu_s_per_m is not None else defaults["mu_s_per_m"]
        self._g = g if g is not None else defaults["g"]
        self._n = n if n is not None else defaults["n"]
        self.oil_type = oil_type
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
