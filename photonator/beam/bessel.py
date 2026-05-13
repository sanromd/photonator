"""Bessel beam profile (J0 radial amplitude).

Non-diffracting Bessel beam at the aperture plane. Photon positions are
sampled from the J0^2 intensity profile via rejection sampling. The cone
angle theta_cone_rad controls the axial wavevector component.

Extension hook: emission_wavelength_nm for wideband/spectral sources.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0  # type: ignore[import]

from photonator.beam.base import AbstractBeam
from photonator.core.photon import PhotonBatch


class BesselBeam(AbstractBeam):
    """Zeroth-order Bessel beam J0(k_r * r).

    Parameters
    ----------
    k_r_per_m : radial wavevector (m^-1); controls ring spacing
    theta_cone_rad : cone half-angle (rad); sets uz = cos(theta_cone)
    aperture_radius_m : maximum radial extent sampled (m)
    emission_wavelength_nm : stub for spectral extension (unused)
    """

    def __init__(
        self,
        k_r_per_m: float,
        theta_cone_rad: float,
        aperture_radius_m: float,
        emission_wavelength_nm: float | None = None,
    ) -> None:
        self.k_r_per_m = k_r_per_m
        self.theta_cone_rad = theta_cone_rad
        self.aperture_radius_m = aperture_radius_m
        self.emission_wavelength_nm = emission_wavelength_nm  # future spectral hook

    def initialize(self, n_photons: int, rng: np.random.Generator) -> PhotonBatch:
        """Sample photon positions from J0^2 intensity via rejection sampling."""
        r_max = self.aperture_radius_m
        r_grid = np.linspace(0.0, r_max, 10000)
        i_grid = j0(self.k_r_per_m * r_grid) ** 2
        peak = max(float(i_grid.max()), 1e-30)

        accepted_r = np.empty(n_photons)
        n_done = 0
        while n_done < n_photons:
            needed = n_photons - n_done
            r_try = rng.uniform(0.0, r_max, needed * 5)
            accept_prob = j0(self.k_r_per_m * r_try) ** 2 / peak
            u = rng.random(len(r_try))
            accepted = r_try[u < accept_prob]
            take = min(len(accepted), needed)
            accepted_r[n_done : n_done + take] = accepted[:take]
            n_done += take

        phi = rng.uniform(0.0, 2.0 * np.pi, n_photons)
        x_m = accepted_r * np.cos(phi)
        y_m = accepted_r * np.sin(phi)

        uz = np.full(n_photons, np.cos(self.theta_cone_rad))
        sin_cone = np.sin(self.theta_cone_rad)
        ux = sin_cone * np.cos(phi)
        uy = sin_cone * np.sin(phi)

        batch = PhotonBatch(n_photons, rng=rng)
        batch.x_m[:] = x_m
        batch.y_m[:] = y_m
        batch.ux[:] = ux
        batch.uy[:] = uy
        batch.uz[:] = uz
        return batch
