"""Henyey-Greenstein phase function with analytical CDF inversion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from photonator.phase_functions.base import AbstractPhaseFunction


class HenyeyGreensteinPhaseFunction(AbstractPhaseFunction):
    """Henyey-Greenstein phase function p(θ; g).

    p(θ) = (1 - g²) / (4π * (1 + g² - 2g·cos θ)^(3/2))

    The CDF admits an analytical inverse:
        cos θ = (1/(2g)) * [1 + g² - ((1-g²)/(1-g+2g·U))²]   for g ≠ 0
        cos θ = 1 - 2U                                          for g = 0

    Parameters
    ----------
    g : asymmetry parameter in (-1, 1).  g=0 → isotropic, g>0 → forward scattering.
    """

    def __init__(self, g: float) -> None:
        if not -1.0 < g < 1.0:
            raise ValueError(f"g must be in (-1, 1), got {g}")
        self.g = g

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Sample n polar scattering angles via analytical CDF inversion."""
        u = rng.random(n)
        g = self.g
        if abs(g) < 1e-12:
            # Isotropic: cos θ ~ Uniform(-1, 1)
            cos_theta = 1.0 - 2.0 * u
        else:
            # HG analytical inverse
            term = (1.0 - g**2) / (1.0 - g + 2.0 * g * u)
            cos_theta = (1.0 + g**2 - term**2) / (2.0 * g)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        return theta.astype(np.float64)
