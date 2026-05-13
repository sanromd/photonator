"""Gaussian beam profile initializer.

Faithfully translates beamProfile.m: Rayleigh-CDF radius sampling combined
with a thin-lens ray-matrix transformation to produce the divergence angle.
"""

from __future__ import annotations

import numpy as np

from photonator.beam.base import AbstractBeam
from photonator.core.photon import PhotonBatch, X, Y, UX, UY, UZ


class GaussianBeam(AbstractBeam):
    """Paraxial Gaussian beam with thin-lens divergence.

    The transverse radius is sampled from the Rayleigh distribution
    (i.e. r ~ sqrt(-log(1-U)) * w0_m) and the divergence angle is
    applied via the thin-lens ray matrix (theta_div = (diverg/w0) * r).

    Parameters
    ----------
    w0_m : 1/e beam half-width at waist (m)
    half_angle_divergence_rad : half-angle beam divergence (rad)
    """

    def __init__(self, w0_m: float, half_angle_divergence_rad: float = 0.0) -> None:
        self.w0_m = w0_m
        self.half_angle_divergence_rad = half_angle_divergence_rad

    def initialize(self, n_photons: int, rng: np.random.Generator) -> PhotonBatch:
        """Initialise photon positions and directions for a Gaussian beam."""
        u = rng.random(n_photons)
        radius_m = self.w0_m * np.sqrt(-np.log(np.maximum(1.0 - u, 1e-15)))

        inv_f = -self.half_angle_divergence_rad / self.w0_m if self.w0_m > 0 else 0.0
        div_ang = -inv_f * radius_m  # polar divergence angle (rad)

        phi = rng.uniform(0.0, 2.0 * np.pi, n_photons)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        uz = np.cos(div_ang)
        sin_div = np.sqrt(np.maximum(1.0 - uz**2, 0.0))
        ux = sin_div * cos_phi
        uy = sin_div * sin_phi

        x_m = radius_m * cos_phi
        y_m = radius_m * sin_phi

        batch = PhotonBatch(n_photons, rng=rng)
        batch.x_m[:] = x_m
        batch.y_m[:] = y_m
        batch.ux[:] = ux
        batch.uy[:] = uy
        batch.uz[:] = uz
        return batch
