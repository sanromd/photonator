"""Abstract base class for optical media.

Extension stubs for spectral (wideband) and inelastic scattering are
included as optional attributes so future subclasses can add them
without changing the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AbstractMedium(ABC):
    """Optical medium providing absorption and scattering coefficients.

    Wideband / inelastic extension stubs
    -------------------------------------
    wavelength_nm : float | None
        Operating wavelength. When set, subclasses may interpolate
        wavelength-dependent mu_a and mu_s from a spectral table.
    emission_wavelength_nm : float | None
        Re-emission wavelength for fluorescence or Raman.  Set on the
        medium so the propagation loop can redirect emitted photons
        to a different wavelength channel.  Currently unused.
    inelastic_yield : float | None
        Fraction of scattered power that is inelastically re-emitted.
        Currently unused; reserved for future wideband extension.
    """

    # Spectral / inelastic extension stubs
    wavelength_nm: float | None = None
    emission_wavelength_nm: float | None = None
    inelastic_yield: float | None = None

    @property
    @abstractmethod
    def mu_a_per_m(self) -> float:
        """Absorption coefficient (m^-1)."""

    @property
    @abstractmethod
    def mu_s_per_m(self) -> float:
        """Scattering coefficient (m^-1)."""

    @property
    @abstractmethod
    def g(self) -> float:
        """Scattering asymmetry parameter (dimensionless, -1 < g < 1)."""

    @property
    @abstractmethod
    def n(self) -> float:
        """Refractive index (dimensionless)."""

    @property
    def mu_t_per_m(self) -> float:
        """Total attenuation coefficient c = a + b (m^-1)."""
        return self.mu_a_per_m + self.mu_s_per_m

    @property
    def albedo(self) -> float:
        """Single-scatter albedo b/c."""
        c = self.mu_t_per_m
        return self.mu_s_per_m / c if c > 0 else 0.0

    @classmethod
    def from_file(cls, path: Path | str) -> "TabulatedMedium":
        """Load a custom medium from an HDF5 or CSV spectral file.

        The file must contain columns/datasets:
        wavelength_nm, mu_a_per_m, mu_s_per_m, g, n

        Returns a TabulatedMedium that interpolates to any wavelength.
        """
        return TabulatedMedium.load(Path(path))


class TabulatedMedium(AbstractMedium):
    """Medium whose optical properties are interpolated from a spectral table.

    Parameters
    ----------
    wavelengths_nm : 1-D array of wavelengths (nm)
    mu_a_table : absorption coefficients (m^-1), shape matches wavelengths_nm
    mu_s_table : scattering coefficients (m^-1)
    g_table : asymmetry parameters
    n_table : refractive indices
    wavelength_nm : query wavelength (nm)
    """

    def __init__(
        self,
        wavelengths_nm: np.ndarray,
        mu_a_table: np.ndarray,
        mu_s_table: np.ndarray,
        g_table: np.ndarray,
        n_table: np.ndarray,
        wavelength_nm: float,
    ) -> None:
        self._wl_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        self._mu_a = np.asarray(mu_a_table, dtype=np.float64)
        self._mu_s = np.asarray(mu_s_table, dtype=np.float64)
        self._g = np.asarray(g_table, dtype=np.float64)
        self._n = np.asarray(n_table, dtype=np.float64)
        self.wavelength_nm = wavelength_nm

    def _interp(self, table: np.ndarray) -> float:
        return float(np.interp(self.wavelength_nm, self._wl_nm, table))

    @property
    def mu_a_per_m(self) -> float:
        return self._interp(self._mu_a)

    @property
    def mu_s_per_m(self) -> float:
        return self._interp(self._mu_s)

    @property
    def g(self) -> float:
        return self._interp(self._g)

    @property
    def n(self) -> float:
        return self._interp(self._n)

    @classmethod
    def load(cls, path: Path) -> "TabulatedMedium":
        """Load from HDF5 or CSV; wavelength_nm defaults to first entry."""
        if path.suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as f:
                wl = np.array(f["wavelength_nm"])
                mu_a = np.array(f["mu_a_per_m"])
                mu_s = np.array(f["mu_s_per_m"])
                g = np.array(f["g"])
                n = np.array(f["n"])
        elif path.suffix == ".csv":
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            wl, mu_a, mu_s, g, n = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return cls(wl, mu_a, mu_s, g, n, wavelength_nm=float(wl[0]))
