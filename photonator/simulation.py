"""Simulation: top-level MC orchestrator class.

Faithfully replicates the batch loop in photon_sim_r4.m, extended with
pluggable beam, medium, phase function, and GPU backend support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from photonator.beam.base import AbstractBeam
from photonator.core.propagation import propagate_cpu
from photonator.core.receiver import Receiver
from photonator.media.base import AbstractMedium
from photonator.phase_functions.base import AbstractPhaseFunction


Backend = Literal["cpu", "numba", "cupy"]


@dataclass
class SimulationResult:
    """Aggregated output of a completed Simulation.

    All statistics are accumulated across all batches.
    """

    # Scalar statistics
    total_power: float = 0.0        # sum of received photon weights (after Fresnel/FOV)
    total_packets: int = 0          # number of detected photons (passed all filters)
    angle_mean_rad: float = 0.0     # mean uz (cos of arrival angle)
    angle_var: float = 0.0
    dist_mean_m: float = 0.0        # mean path length of received photons
    dist_var: float = 0.0
    weight_mean: float = 0.0        # normalized: total_power / n_photons
    reflected: int = 0              # photons rejected at critical angle

    # Run metadata
    n_photons: int = 0              # total photons launched (n_photons * n_batches)
    n_batches: int = 0
    elapsed_s: float = 0.0

    # Raw photon data (None unless store_photons=True)
    rec_loc: NDArray[np.float64] | None = None
    distances_m: NDArray[np.float64] | None = None
    rec_weights: NDArray[np.float64] | None = None


class Simulation:
    """Monte Carlo photon transport simulation.

    Parameters
    ----------
    medium : optical medium (AbstractMedium subclass)
    beam : beam profile initializer (AbstractBeam subclass)
    phase_fn : phase function (AbstractPhaseFunction subclass)
    receiver : Receiver instance
    n_photons : photons per batch
    n_batches : number of independent batches to run
    backend : "cpu" (NumPy), "numba" (CUDA JIT), or "cupy" (CuPy vectorized)
    store_photons : if True, concatenate raw photon data in the result
    seed : master RNG seed; each batch gets seed + batch_index
    """

    def __init__(
        self,
        medium: AbstractMedium,
        beam: AbstractBeam,
        phase_fn: AbstractPhaseFunction,
        receiver: Receiver,
        n_photons: int = 100_000,
        n_batches: int = 1,
        backend: Backend = "cpu",
        store_photons: bool = False,
        seed: int = 0,
    ) -> None:
        self.medium = medium
        self.beam = beam
        self.phase_fn = phase_fn
        self.receiver = receiver
        self.n_photons = n_photons
        self.n_batches = n_batches
        self.backend = backend
        self.store_photons = store_photons
        self.seed = seed

    def run(self) -> SimulationResult:
        """Run all batches and return aggregated SimulationResult."""
        t0 = time.perf_counter()
        self.receiver.reset()

        all_rec_loc: list[NDArray] = []
        all_dist: list[NDArray] = []
        all_weights: list[NDArray] = []

        total_n_tx = self.n_photons * self.n_batches

        for batch_idx in range(self.n_batches):
            rng = np.random.default_rng(self.seed + batch_idx)
            batch = self.beam.initialize(self.n_photons, rng)

            if self.backend == "cpu":
                _elapsed, rec_loc, distances_m, rec_weights, _n_packets = propagate_cpu(
                    batch, self.medium, self.phase_fn, self.receiver
                )
            elif self.backend == "cupy":
                from photonator.gpu.propagation_cupy import propagate_cupy
                _elapsed, rec_loc, distances_m, rec_weights, _n_packets = propagate_cupy(
                    batch, self.medium, self.phase_fn, self.receiver
                )
            elif self.backend == "numba":
                from photonator.gpu.propagation_numba import propagate_numba
                cdf, angles_rad = _get_cdf(self.phase_fn)
                _elapsed, rec_loc, distances_m, rec_weights, _n_packets = propagate_numba(
                    batch, self.medium, cdf, angles_rad, self.receiver,
                    seed=self.seed + batch_idx,
                )
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")

            # Apply Fresnel, FOV, aperture filtering and accumulate Welford stats
            self.receiver.detect(rec_loc, distances_m, rec_weights, total_n_tx)

            if self.store_photons and len(rec_loc):
                all_rec_loc.append(rec_loc)
                all_dist.append(distances_m)
                all_weights.append(rec_weights)

        elapsed_s = time.perf_counter() - t0

        n_det = self.receiver._count
        result = SimulationResult(
            total_power=self.receiver._power,
            total_packets=n_det,
            angle_mean_rad=self.receiver._angle_mean,
            angle_var=self.receiver._angle_M2 / max(n_det - 1, 1),
            dist_mean_m=self.receiver._dist_mean,
            dist_var=self.receiver._dist_M2 / max(n_det - 1, 1),
            weight_mean=self.receiver._power / max(total_n_tx, 1),
            reflected=self.receiver._reflected,
            n_photons=total_n_tx,
            n_batches=self.n_batches,
            elapsed_s=elapsed_s,
        )

        if self.store_photons and all_rec_loc:
            result.rec_loc = np.concatenate(all_rec_loc)
            result.distances_m = np.concatenate(all_dist)
            result.rec_weights = np.concatenate(all_weights)

        return result


def _get_cdf(
    phase_fn: AbstractPhaseFunction,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the CDF and angle array from a phase function (for Numba kernel)."""
    if hasattr(phase_fn, "_cdf") and hasattr(phase_fn, "_angles_rad"):
        return phase_fn._cdf, phase_fn._angles_rad
    # Build from HG formula
    g = getattr(phase_fn, "g", 0.0)
    theta = np.concatenate([np.arange(0, 10, 0.01), np.arange(10.1, 180.1, 0.1)]) * np.pi / 180.0
    gsqr = g**2
    vsf = (1.0 - gsqr) / (4.0 * np.pi * (1.0 + gsqr - 2.0 * g * np.cos(theta)) ** 1.5)
    return phase_fn.build_cdf(theta, vsf)
