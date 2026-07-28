"""Layered medium: hard-boundary, gradient, and random-spatial layer types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from photonator.media.base import AbstractMedium


@dataclass
class Layer:
    """A single axial layer with hard boundaries.

    Parameters
    ----------
    z_start_m : start of layer (m, inclusive)
    z_end_m : end of layer (m, exclusive for all but the last layer)
    medium : optical medium for this layer
    """

    z_start_m: float
    z_end_m: float
    medium: AbstractMedium


class LayeredMedium(AbstractMedium):
    """Axially stratified medium composed of discrete layers.

    Layer boundaries are hard: optical properties switch instantly at
    the interface, and Snell's law + Fresnel are applied on crossing
    (handled by the propagation loop, not here).

    The medium properties returned by this class correspond to the
    layer at z = query_z_m.  Use :meth:`layer_at` to look up the
    layer for an arbitrary z position.

    Parameters
    ----------
    layers : list of Layer objects, ordered by z_start_m (ascending)
    """

    def __init__(self, layers: list[Layer]) -> None:
        if not layers:
            raise ValueError("LayeredMedium requires at least one layer")
        layers = sorted(layers, key=lambda l: l.z_start_m)
        self._layers = layers
        self._z_starts = np.array([l.z_start_m for l in layers])

    def layer_at(self, z_m: float) -> Layer:
        """Return the layer containing axial position z_m."""
        idx = int(np.searchsorted(self._z_starts, z_m, side="right")) - 1
        idx = max(0, min(idx, len(self._layers) - 1))
        return self._layers[idx]

    def _current_medium(self) -> AbstractMedium:
        """Return the medium at the query_z_m set on this instance."""
        z = getattr(self, "_query_z_m", self._layers[0].z_start_m)
        return self.layer_at(z).medium

    @property
    def mu_a_per_m(self) -> float:
        return self._current_medium().mu_a_per_m

    @property
    def mu_s_per_m(self) -> float:
        return self._current_medium().mu_s_per_m

    @property
    def g(self) -> float:
        return self._current_medium().g

    @property
    def n(self) -> float:
        return self._current_medium().n

    def set_query_z(self, z_m: float) -> LayeredMedium:
        """Set the axial position for property queries (returns self)."""
        self._query_z_m = z_m
        return self

    def boundary_crossings(
        self, z0_m: float, z1_m: float
    ) -> list[tuple[float, Layer, Layer]]:
        """Return list of (z_boundary, layer_before, layer_after) crossings.

        Used by the propagation loop to apply Fresnel at each interface.
        """
        crossings = []
        for layer in self._layers[1:]:
            zb = layer.z_start_m
            if z0_m < zb <= z1_m:
                before = self.layer_at(zb - 1e-12)
                after = self.layer_at(zb + 1e-12)
                crossings.append((zb, before, after))
        return crossings


class GradientMedium(AbstractMedium):
    """Medium whose properties vary smoothly with depth via user-supplied functions.

    Parameters
    ----------
    mu_a_func : callable(z_m: float) → mu_a (m^-1)
    mu_s_func : callable(z_m: float) → mu_s (m^-1)
    g_func : callable(z_m: float) → g (dimensionless)
    n_func : callable(z_m: float) → n (dimensionless)
    query_z_m : axial position for current queries
    """

    def __init__(
        self,
        mu_a_func: Callable[[float], float],
        mu_s_func: Callable[[float], float],
        g_func: Callable[[float], float],
        n_func: Callable[[float], float],
        query_z_m: float = 0.0,
    ) -> None:
        self._mu_a_func = mu_a_func
        self._mu_s_func = mu_s_func
        self._g_func = g_func
        self._n_func = n_func
        self._query_z_m = query_z_m

    def set_query_z(self, z_m: float) -> GradientMedium:
        self._query_z_m = z_m
        return self

    @property
    def mu_a_per_m(self) -> float:
        return self._mu_a_func(self._query_z_m)

    @property
    def mu_s_per_m(self) -> float:
        return self._mu_s_func(self._query_z_m)

    @property
    def g(self) -> float:
        return self._g_func(self._query_z_m)

    @property
    def n(self) -> float:
        return self._n_func(self._query_z_m)
