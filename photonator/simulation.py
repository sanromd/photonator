"""Simulation: top-level MC orchestrator class.

Faithfully replicates the batch loop in photon_sim_r4.m, extended with
pluggable beam, medium, phase function, and GPU backend support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from photonator.beam.base import AbstractBeam
from photonator.core.photon import PhotonBatch
from photonator.core.propagation import propagate_cpu
from photonator.core.propagation_layered import propagate_layered_cpu
from photonator.core.receiver import Receiver
from photonator.media.base import AbstractMedium
from photonator.media.layers import GradientMedium, LayeredMedium
from photonator.phase_functions.base import AbstractPhaseFunction

Backend = Literal["cpu", "numba", "cupy"]


def _fluorescence_active(medium: AbstractMedium) -> bool:
    """True when the medium carries an active fluorophore (Phase 3 hooks)."""
    qy = getattr(medium, "inelastic_yield", None) or 0.0
    mu_a_f = getattr(medium, "mu_a_fluorophore_per_m", 0.0) or 0.0
    return qy > 0.0 and mu_a_f > 0.0 and getattr(medium, "emission_wavelength_nm", None) is not None


def _make_emission_batch(
    src: PhotonBatch, mask: NDArray[np.bool_], rng: np.random.Generator
) -> PhotonBatch:
    """Build the emission-pass batch from fluoresced photons.

    Positions, weights, and accumulated path lengths carry over from the
    conversion points; directions are re-drawn isotropically (fluorescence
    emission has no angular memory of the excitation photon).
    """
    m = int(np.sum(mask))
    b = PhotonBatch(m, rng=rng)
    b.x_m[:] = src.x_m[mask]
    b.y_m[:] = src.y_m[mask]
    b.z_m[:] = src.z_m[mask]
    b.weight[:] = src.weight[mask]
    b._path_length_m[:] = src._path_length_m[mask]

    uz = rng.uniform(-1.0, 1.0, m)
    phi = rng.uniform(0.0, 2.0 * np.pi, m)
    sin_polar = np.sqrt(np.maximum(1.0 - uz**2, 0.0))
    b.ux[:] = sin_polar * np.cos(phi)
    b.uy[:] = sin_polar * np.sin(phi)
    b.uz[:] = uz
    return b


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

    # Fluorescence channel (Phase 3; zero unless the medium has active hooks)
    fluorescent_power: float = 0.0      # detected power at the emission wavelength
    fluorescent_packets: int = 0        # detected fluorescence photons
    fluoresced_photons: int = 0         # photons converted (detected or not)
    emission_wavelength_nm: float | None = None

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

        if isinstance(self.medium, GradientMedium):
            raise NotImplementedError(
                "GradientMedium propagation (adaptive sub-stepping) is not yet wired "
                "into the loop; discretise the gradient into a LayeredMedium instead."
            )
        is_layered = isinstance(self.medium, LayeredMedium)
        fluor_active = _fluorescence_active(self.medium)
        if (is_layered or fluor_active) and self.backend != "cpu":
            raise NotImplementedError(
                "Layered and fluorescent propagation are CPU-only for now; "
                'use backend="cpu".'
            )
        fluor_receiver = self.receiver.clone() if fluor_active else None
        n_fluoresced = 0

        all_rec_loc: list[NDArray] = []
        all_dist: list[NDArray] = []
        all_weights: list[NDArray] = []

        total_n_tx = self.n_photons * self.n_batches

        for batch_idx in range(self.n_batches):
            rng = np.random.default_rng(self.seed + batch_idx)
            batch = self.beam.initialize(self.n_photons, rng)

            if self.backend == "cpu":
                propagate = propagate_layered_cpu if is_layered else propagate_cpu
                _elapsed, rec_loc, distances_m, rec_weights, _n_packets = propagate(
                    batch, self.medium, self.phase_fn, self.receiver
                )
            elif self.backend == "cupy":
                from photonator.gpu.propagation_cupy import propagate_cupy
                _elapsed, rec_loc, distances_m, rec_weights, _n_packets = propagate_cupy(
                    batch, self.medium, self.phase_fn, self.receiver
                )
            elif self.backend == "numba":
                from photonator.gpu.propagation_numba import propagate_numba
                cdf, angles_rad = self.phase_fn.cdf_table()
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

            # Emission pass: re-propagate fluoresced photons isotropically
            # in the host medium at the emission wavelength (single
            # generation — re-absorption by the fluorophore is neglected).
            if fluor_active:
                fl_mask = batch.fluoresced_mask
                n_fl = int(np.sum(fl_mask))
                n_fluoresced += n_fl
                if n_fl > 0:
                    emission_batch = _make_emission_batch(batch, fl_mask, rng)
                    emission_medium = (
                        self.medium.emission_medium()
                        if hasattr(self.medium, "emission_medium")
                        else self.medium
                    )
                    _e2, rec_loc2, dist2, w2, _p2 = propagate_cpu(
                        emission_batch, emission_medium, self.phase_fn, self.receiver
                    )
                    fluor_receiver.detect(rec_loc2, dist2, w2, total_n_tx)

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
            fluorescent_power=fluor_receiver._power if fluor_active else 0.0,
            fluorescent_packets=fluor_receiver._count if fluor_active else 0,
            fluoresced_photons=n_fluoresced,
            emission_wavelength_nm=(
                getattr(self.medium, "emission_wavelength_nm", None) if fluor_active else None
            ),
            n_photons=total_n_tx,
            n_batches=self.n_batches,
            elapsed_s=elapsed_s,
        )

        if self.store_photons and all_rec_loc:
            result.rec_loc = np.concatenate(all_rec_loc)
            result.distances_m = np.concatenate(all_dist)
            result.rec_weights = np.concatenate(all_weights)

        return result
