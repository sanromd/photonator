"""Airy beam profile (2D separable Ai(x/x0)*Ai(y/y0) amplitude).

Photon positions are sampled from the 2D Airy intensity profile
|Ai(x/x0)|^2 * |Ai(y/y0)|^2 via rejection sampling on a finite grid.
Propagation is collimated (uz=1). Divergence extension follows the same
thin-lens approach as GaussianBeam.

Extension hook: emission_wavelength_nm for wideband/spectral sources.
"""

from __future__ import annotations

import numpy as np
from scipy.special import airy  # type: ignore[import]

from photonator.beam.base import AbstractBeam
from photonator.core.photon import PhotonBatch


class AiryBeam(AbstractBeam):
    """2D separable Airy beam Ai(x/x0) * Ai(y/y0).

    Parameters
    ----------
    x0_m : transverse scale in x (m)
    y0_m : transverse scale in y (m)
    extent_x0 : sampling extent in units of x0 (default 6 → ±3*x0)
    extent_y0 : sampling extent in units of y0
    emission_wavelength_nm : stub for spectral extension (unused)
    """

    def __init__(
        self,
        x0_m: float,
        y0_m: float,
        extent_x0: float = 6.0,
        extent_y0: float = 6.0,
        emission_wavelength_nm: float | None = None,
    ) -> None:
        self.x0_m = x0_m
        self.y0_m = y0_m
        self.x_lim_m = extent_x0 * x0_m
        self.y_lim_m = extent_y0 * y0_m
        self.emission_wavelength_nm = emission_wavelength_nm  # future spectral hook

    def _intensity_1d(self, vals: np.ndarray, scale: float) -> np.ndarray:
        ai, *_ = airy(vals / scale)
        return ai**2

    def initialize(self, n_photons: int, rng: np.random.Generator) -> PhotonBatch:
        """Sample 2D positions from separable Airy intensity via rejection."""
        x_max, y_max = self.x_lim_m, self.y_lim_m

        # Peak for envelope
        xi = np.linspace(-x_max, x_max, 5000)
        yi = np.linspace(-y_max, y_max, 5000)
        peak_x = float(self._intensity_1d(xi, self.x0_m).max())
        peak_y = float(self._intensity_1d(yi, self.y0_m).max())
        peak = peak_x * peak_y

        accepted_x = np.empty(n_photons)
        accepted_y = np.empty(n_photons)
        n_done = 0
        while n_done < n_photons:
            needed = n_photons - n_done
            x_try = rng.uniform(-x_max, x_max, needed * 4)
            y_try = rng.uniform(-y_max, y_max, needed * 4)
            accept_prob = (
                self._intensity_1d(x_try, self.x0_m)
                * self._intensity_1d(y_try, self.y0_m)
                / peak
            )
            u = rng.random(len(x_try))
            ok = u < accept_prob
            take = min(int(np.sum(ok)), needed)
            accepted_x[n_done : n_done + take] = x_try[ok][:take]
            accepted_y[n_done : n_done + take] = y_try[ok][:take]
            n_done += take

        batch = PhotonBatch(n_photons, rng=rng)
        batch.x_m[:] = accepted_x
        batch.y_m[:] = accepted_y
        batch.ux[:] = 0.0
        batch.uy[:] = 0.0
        batch.uz[:] = 1.0
        return batch
