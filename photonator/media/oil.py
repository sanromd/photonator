"""Oil optical model (crude and refined petroleum products).

IOR values from:
  Millán-Merino et al. (2020), Energies, for crude oils ~1.46–1.52
  Refined mineral oil: ~1.46 at 589 nm

Absorption is highly variable; conservative estimates used as defaults.

DispersiveOil extends Oil with a Cauchy wavelength-dependent IOR and an
optional numerical Kramers-Kronig computation of n(λ) from a tabulated
imaginary-index k(λ) spectrum.
"""

from __future__ import annotations

import numpy as np

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


# ── Cauchy coefficients calibrated to literature n at 589 nm ─────────────────
# n(λ_μm) = A + B/λ² + C/λ⁴    (λ in micrometers)
_CAUCHY: dict[str, dict[str, float]] = {
    "crude":   {"A": 1.4829, "B": 0.0060, "C": 2.5e-5},  # n(589 nm) ≈ 1.50
    "refined": {"A": 1.4471, "B": 0.0045, "C": 2.2e-5},  # n(589 nm) ≈ 1.46
    "mineral": {"A": 1.4483, "B": 0.0040, "C": 2.0e-5},  # n(589 nm) ≈ 1.46
}


class DispersiveOil(Oil):
    """Oil with Cauchy (Sellmeier) wavelength-dependent refractive index.

    Uses the single-term Cauchy formula:

        n(λ) = A + B / λ_μm² + C / λ_μm⁴

    with coefficients calibrated so n(589 nm) matches the literature values
    in ``_OIL_PARAMS``.  Optionally replaces this with a numerical
    Kramers-Kronig (KK) integral when a measured imaginary-index spectrum
    is supplied.

    Parameters
    ----------
    oil_type : "crude", "refined", or "mineral"
    wavelength_nm : operating wavelength (nm); required for spectral n
    mu_a_per_m, mu_s_per_m, g : same overrides as ``Oil``
    k_wavelengths_nm : 1-D array of wavelengths (nm) for KK integral
    k_spectrum : imaginary index k(λ) at k_wavelengths_nm; enables KK
    """

    def __init__(
        self,
        oil_type: str = "refined",
        wavelength_nm: float | None = None,
        mu_a_per_m: float | None = None,
        mu_s_per_m: float | None = None,
        g: float | None = None,
        k_wavelengths_nm: np.ndarray | None = None,
        k_spectrum: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            oil_type=oil_type,
            mu_a_per_m=mu_a_per_m,
            mu_s_per_m=mu_s_per_m,
            g=g,
        )
        self.wavelength_nm = wavelength_nm
        self._cauchy = _CAUCHY[oil_type]

        # KK tables (optional)
        if (k_wavelengths_nm is None) != (k_spectrum is None):
            raise ValueError("Provide both k_wavelengths_nm and k_spectrum, or neither.")
        self._kk_wl = (
            np.asarray(k_wavelengths_nm, dtype=np.float64) if k_wavelengths_nm is not None else None
        )
        self._kk_k = (
            np.asarray(k_spectrum, dtype=np.float64) if k_spectrum is not None else None
        )

    @property
    def n(self) -> float:
        if self.wavelength_nm is None:
            return self._n   # fall back to fixed value from Oil
        if self._kk_wl is not None:
            return self._kramers_kronig_n(self.wavelength_nm)
        return self._cauchy_n(self.wavelength_nm)

    def n_at(self, wavelength_nm: float) -> float:
        """Evaluate n at an arbitrary wavelength without changing state."""
        if self._kk_wl is not None:
            return self._kramers_kronig_n(wavelength_nm)
        return self._cauchy_n(wavelength_nm)

    def _cauchy_n(self, wavelength_nm: float) -> float:
        lam_um = wavelength_nm / 1000.0
        c = self._cauchy
        return c["A"] + c["B"] / lam_um**2 + c["C"] / lam_um**4

    def _kramers_kronig_n(self, query_nm: float) -> float:
        """Numerical Kramers-Kronig integral in angular-frequency space.

        n(ω₀) = 1 + (2/π) P∫₀^∞ ω k(ω) / (ω² − ω₀²) dω

        The Cauchy principal value is approximated by excluding the bin
        that contains the singularity.
        """
        c_light = 2.997_924_58e8  # m/s
        wl_m = self._kk_wl * 1e-9
        omega = 2.0 * np.pi * c_light / wl_m         # rad/s, descending
        omega_0 = 2.0 * np.pi * c_light / (query_nm * 1e-9)

        # Sort ascending
        idx = np.argsort(omega)
        omega = omega[idx]
        k = self._kk_k[idx]

        # Mask the singular bin
        dw = omega[1] - omega[0]
        mask = np.abs(omega - omega_0) > 0.5 * dw
        integrand = omega[mask] * k[mask] / (omega[mask] ** 2 - omega_0**2)
        n_delta = (2.0 / np.pi) * float(np.trapezoid(integrand, omega[mask]))
        return 1.0 + n_delta
