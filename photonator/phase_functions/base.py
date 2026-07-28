"""Abstract base class for phase functions.

Extension hooks for inelastic/wideband scattering are included as
optional attributes with None defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class AbstractPhaseFunction(ABC):
    """Samples polar scattering angles θ from a given phase function.

    Inelastic extension stubs
    -------------------------
    emission_wavelength_nm : float | None
        If set, sampled photons should be re-emitted at this wavelength
        (fluorescence / Raman).  Currently unused; reserved for future work.
    inelastic_yield : float | None
        Probability per scattering event that the photon undergoes an
        inelastic process.  Currently unused; reserved for future work.
    """

    # -- Inelastic extension stubs (unused in Phase 1–4) --
    emission_wavelength_nm: float | None = None
    inelastic_yield: float | None = None

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw n polar scattering angles θ (rad) from the phase function.

        Parameters
        ----------
        n : number of angles to sample
        rng : NumPy random generator

        Returns
        -------
        theta : float64 array of shape (n,), values in [0, π]
        """

    def cdf_table(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (cdf, angles_rad) for tabulated inverse-transform sampling.

        The default implementation returns the ``_cdf`` / ``_angles_rad``
        arrays built at construction time by tabulated phase functions
        (Petzold, Mie).  Analytical phase functions should override this
        to build a dense table on demand — the GPU backends need a table
        even when CPU sampling is analytical.

        Raises
        ------
        NotImplementedError
            If the phase function has no tabulated CDF and does not
            override this method.
        """
        cdf = getattr(self, "_cdf", None)
        angles = getattr(self, "_angles_rad", None)
        if cdf is None or angles is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no tabulated CDF; override cdf_table()."
            )
        return cdf, angles

    def build_cdf(
        self,
        angles_rad: NDArray[np.float64],
        vsf: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build a normalised CDF from a tabulated VSF.

        Integrates vsf(θ)*sin(θ) via cumulative trapezoidal rule and
        normalises to [0, 1].  Matches generate_scatter.m exactly.

        Parameters
        ----------
        angles_rad : angle grid (rad), shape (K,)
        vsf : volume scattering function values, shape (K,)

        Returns
        -------
        cdf : normalised cumulative distribution, shape (K,)
        angles_rad : unchanged (returned for convenience)
        """
        from scipy.integrate import cumulative_trapezoid  # type: ignore[import]

        integrand = vsf * np.sin(angles_rad)
        cdf = cumulative_trapezoid(integrand, angles_rad, initial=0.0)
        cdf_max = cdf[-1]
        if cdf_max > 0:
            cdf /= cdf_max
        return cdf, angles_rad

    def sample_from_cdf(
        self,
        cdf: NDArray[np.float64],
        angles_rad: NDArray[np.float64],
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Inverse-transform sample θ from a pre-built CDF.

        Uses np.searchsorted + linear interpolation, replacing MATLAB's
        binary-search lookupCDF.m.
        """
        u = rng.random(n)
        k = np.searchsorted(cdf, u, side="right")
        k = np.clip(k, 1, len(cdf) - 1)
        # Linear interpolation between bracketing CDF values
        cdf_lo = cdf[k - 1]
        cdf_hi = cdf[k]
        ang_lo = angles_rad[k - 1]
        ang_hi = angles_rad[k]
        denom = cdf_hi - cdf_lo
        # Avoid division by zero at plateau regions
        safe_denom = np.where(denom > 1e-15, denom, 1.0)
        frac = np.where(denom > 1e-15, (u - cdf_lo) / safe_denom, 0.0)
        theta = ang_lo + frac * (ang_hi - ang_lo)
        return theta.astype(np.float64)
