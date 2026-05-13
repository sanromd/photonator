"""Laguerre-Gaussian beam profile (LG_p^l mode).

Paraxial approximation. Photon positions are sampled from the LG intensity
profile |LG_p^l(r,phi)|^2 using rejection sampling against an envelope,
and the divergence is set to zero (collimated beam at the waist).
Extension to focused / diverging LG beams follows the same thin-lens
approach as GaussianBeam.

Extension hook: set emission_wavelength_nm to enable spectral tracking.
"""

from __future__ import annotations

import numpy as np
from scipy.special import genlaguerre  # type: ignore[import]

from photonator.beam.base import AbstractBeam
from photonator.core.photon import PhotonBatch


class LaguerreGaussianBeam(AbstractBeam):
    """Laguerre-Gaussian LG_p^l beam at the waist (z=0).

    Parameters
    ----------
    p : radial index (≥ 0)
    l : azimuthal index (orbital angular momentum)
    w0_m : beam waist radius (m)
    half_angle_divergence_rad : half-angle divergence (rad); 0 = collimated
    emission_wavelength_nm : stub for spectral/inelastic extension (unused)
    """

    def __init__(
        self,
        p: int,
        l: int,
        w0_m: float,
        half_angle_divergence_rad: float = 0.0,
        emission_wavelength_nm: float | None = None,
    ) -> None:
        self.p = p
        self.l = l
        self.w0_m = w0_m
        self.half_angle_divergence_rad = half_angle_divergence_rad
        self.emission_wavelength_nm = emission_wavelength_nm  # future spectral hook
        self._Lp = genlaguerre(p, abs(l))

    def _intensity(self, r_m: np.ndarray) -> np.ndarray:
        """LG intensity profile (unnormalized) as a function of radius."""
        rho = np.sqrt(2.0) * r_m / self.w0_m
        lag = self._Lp(rho**2)
        return (rho ** (2 * abs(self.l))) * (lag**2) * np.exp(-(rho**2))

    def initialize(self, n_photons: int, rng: np.random.Generator) -> PhotonBatch:
        """Sample photon positions from LG intensity via rejection sampling."""
        # Envelope: exponential with scale chosen to exceed LG profile
        r_max = self.w0_m * np.sqrt(2.0 * (2 * self.p + abs(self.l) + 1))
        peak_intensity = self._intensity(np.linspace(0, r_max, 10000)).max()

        accepted_r = np.empty(n_photons)
        n_done = 0
        while n_done < n_photons:
            needed = n_photons - n_done
            r_try = rng.uniform(0.0, r_max, needed * 3)
            accept_prob = self._intensity(r_try) / (peak_intensity + 1e-30)
            u = rng.random(len(r_try))
            accepted = r_try[u < accept_prob]
            take = min(len(accepted), needed)
            accepted_r[n_done : n_done + take] = accepted[:take]
            n_done += take

        phi = rng.uniform(0.0, 2.0 * np.pi, n_photons)
        x_m = accepted_r * np.cos(phi)
        y_m = accepted_r * np.sin(phi)

        inv_f = -self.half_angle_divergence_rad / self.w0_m if self.w0_m > 0 else 0.0
        div_ang = -inv_f * accepted_r
        uz = np.cos(div_ang)
        sin_div = np.sqrt(np.maximum(1.0 - uz**2, 0.0))
        ux = sin_div * np.cos(phi)
        uy = sin_div * np.sin(phi)

        batch = PhotonBatch(n_photons, rng=rng)
        batch.x_m[:] = x_m
        batch.y_m[:] = y_m
        batch.ux[:] = ux
        batch.uy[:] = uy
        batch.uz[:] = uz
        return batch
