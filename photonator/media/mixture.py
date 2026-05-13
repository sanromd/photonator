"""Volume-fraction mixture of two optical media."""

from __future__ import annotations

from photonator.media.base import AbstractMedium


class MixtureMedium(AbstractMedium):
    """Linear volume-fraction mixture of two media.

    Optical coefficients are mixed as:
        mu_x = f * mu_x_a + (1-f) * mu_x_b

    The refractive index uses volume-fraction mixing:
        n = f * n_a + (1-f) * n_b  (linear approximation)

    Parameters
    ----------
    medium_a : first medium
    medium_b : second medium
    volume_fraction_a : volume fraction of medium_a (0–1)
    """

    def __init__(
        self,
        medium_a: AbstractMedium,
        medium_b: AbstractMedium,
        volume_fraction_a: float,
    ) -> None:
        if not 0.0 <= volume_fraction_a <= 1.0:
            raise ValueError("volume_fraction_a must be in [0, 1]")
        self._a = medium_a
        self._b = medium_b
        self._f = volume_fraction_a

    @property
    def mu_a_per_m(self) -> float:
        return self._f * self._a.mu_a_per_m + (1.0 - self._f) * self._b.mu_a_per_m

    @property
    def mu_s_per_m(self) -> float:
        return self._f * self._a.mu_s_per_m + (1.0 - self._f) * self._b.mu_s_per_m

    @property
    def g(self) -> float:
        # Albedo-weighted average of g
        b_a = self._a.mu_s_per_m
        b_b = self._b.mu_s_per_m
        total_b = self._f * b_a + (1.0 - self._f) * b_b
        if total_b < 1e-30:
            return 0.0
        return (self._f * b_a * self._a.g + (1.0 - self._f) * b_b * self._b.g) / total_b

    @property
    def n(self) -> float:
        return self._f * self._a.n + (1.0 - self._f) * self._b.n
