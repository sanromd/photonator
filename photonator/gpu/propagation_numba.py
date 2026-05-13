"""Numba CUDA JIT propagation kernel — one CUDA thread per photon.

Usage::

    from photonator.gpu.propagation_numba import propagate_numba
    elapsed_s, rec_loc, distances_m, rec_weights, n_packets = propagate_numba(
        batch, medium, cdf, angles_rad, receiver
    )

Requires: numba >= 0.57, CUDA toolkit matching the installed numba version.
Falls back silently to CPU if CUDA is unavailable (raises ImportError otherwise).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from photonator.core.photon import PhotonBatch
    from photonator.media.base import AbstractMedium
    from photonator.core.receiver import Receiver

try:
    from numba import cuda, float64, int32
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _make_kernel(inv_c: float, albedo: float, min_w: float, receiver_z: float):
    """JIT-compile and return a CUDA propagation kernel for fixed medium params."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required for the CUDA backend. Install with: pip install numba")

    from numba import cuda
    from numba.cuda.random import xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64

    roulette_inv = 1.0 / 10.0

    @cuda.jit
    def _kernel(photon_state, total_dist, rec_loc, rec_dist, cdf_d, angles_d, rng_states, n_photons):
        """One thread per photon. Runs the full MC loop for a single photon."""
        i = cuda.grid(1)
        if i >= n_photons:
            return

        # STATUS: 1=active, 0=detected, -1=terminated
        if photon_state[i, 7] != 1.0:
            return

        MAX_UZ = 1.0 - 1e-12
        K = len(cdf_d)

        while photon_state[i, 7] == 1.0:
            u1 = xoroshiro128p_uniform_float64(rng_states, i)
            u2 = xoroshiro128p_uniform_float64(rng_states, i)
            u3 = xoroshiro128p_uniform_float64(rng_states, i)

            r = -inv_c * math.log(max(u1, 1e-15))

            # CDF lookup (binary search)
            lo = 1
            hi = K - 1
            while hi > lo:
                mid = (lo + hi) // 2
                if u2 < cdf_d[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            k = lo
            cdf_lo = cdf_d[k - 1]
            cdf_hi = cdf_d[k]
            ang_lo = angles_d[k - 1]
            ang_hi = angles_d[k]
            denom = cdf_hi - cdf_lo
            if denom > 1e-15:
                frac = (u2 - cdf_lo) / denom
            else:
                frac = 0.0
            theta = ang_lo + frac * (ang_hi - ang_lo)
            phi = u3 * 6.283185307179586

            ux = photon_state[i, 3]
            uy = photon_state[i, 4]
            uz = photon_state[i, 5]

            x_step = r * ux
            y_step = r * uy
            z_step = r * uz
            new_z = photon_state[i, 2] + z_step

            if new_z >= receiver_z:
                if abs(uz) > 1e-15:
                    z_dist = receiver_z - photon_state[i, 2]
                    dist_to_rec = z_dist / uz
                    rec_loc[i, 0] = photon_state[i, 0] + z_dist * ux / uz
                    rec_loc[i, 1] = photon_state[i, 1] + z_dist * uy / uz
                    rec_loc[i, 2] = ux
                    rec_loc[i, 3] = uy
                    rec_loc[i, 4] = uz
                    rec_dist[i] = total_dist[i] + dist_to_rec
                    total_dist[i] += dist_to_rec
                photon_state[i, 7] = 0.0  # detected
                break

            # Move
            photon_state[i, 0] += x_step
            photon_state[i, 1] += y_step
            photon_state[i, 2] = new_z
            total_dist[i] += r

            if photon_state[i, 2] < 0.0:
                photon_state[i, 7] = -1.0
                break

            # Implicit absorption
            photon_state[i, 6] *= albedo

            # Roulette
            if photon_state[i, 6] < min_w:
                u_rou = xoroshiro128p_uniform_float64(rng_states, i)
                if u_rou > roulette_inv:
                    photon_state[i, 7] = -1.0
                    break
                else:
                    photon_state[i, 6] *= 10.0

            # Update direction cosines
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            sqrt_1_uz2 = math.sqrt(max(1.0 - uz * uz, 0.0))

            if abs(uz) > MAX_UZ:
                photon_state[i, 3] = sin_t * cos_p
                photon_state[i, 4] = sin_t * sin_p
                photon_state[i, 5] = (1.0 if uz >= 0 else -1.0) * cos_t
            else:
                photon_state[i, 3] = (sin_t / sqrt_1_uz2) * (ux * uz * cos_p - uy * sin_p) + ux * cos_t
                photon_state[i, 4] = (sin_t / sqrt_1_uz2) * (uy * uz * cos_p + ux * sin_p) + uy * cos_t
                photon_state[i, 5] = -sin_t * cos_p * sqrt_1_uz2 + uz * cos_t

            # Normalise
            norm = math.sqrt(photon_state[i, 3]**2 + photon_state[i, 4]**2 + photon_state[i, 5]**2)
            if abs(1.0 - norm) > 1e-11:
                photon_state[i, 3] /= norm
                photon_state[i, 4] /= norm
                photon_state[i, 5] /= norm

    return _kernel


def propagate_numba(
    batch: "PhotonBatch",
    medium: "AbstractMedium",
    cdf: NDArray[np.float64],
    angles_rad: NDArray[np.float64],
    receiver: "Receiver",
    threads_per_block: int = 256,
    seed: int = 42,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Run MC propagation on GPU using Numba CUDA JIT.

    Parameters
    ----------
    batch : PhotonBatch (initialised)
    medium : optical medium
    cdf : pre-built phase-function CDF (shape K,)
    angles_rad : angle grid for CDF (shape K,)
    receiver : Receiver (provides receiver_z_m)
    threads_per_block : CUDA threads per block
    seed : RNG seed for xoroshiro128p states

    Returns
    -------
    elapsed_s, rec_loc, distances_m, rec_weights, n_packets
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")

    from photonator.constants import min_weight_for_albedo

    c_per_m = medium.mu_a_per_m + medium.mu_s_per_m
    albedo = medium.mu_s_per_m / c_per_m if c_per_m > 0 else 0.0
    inv_c = 1.0 / c_per_m
    min_w = min_weight_for_albedo(albedo)

    kernel = _make_kernel(inv_c, albedo, min_w, receiver.receiver_z_m)

    n = batch.n
    state_d = cuda.to_device(batch._state.astype(np.float64))
    total_dist_d = cuda.to_device(np.zeros(n, dtype=np.float64))
    rec_loc_d = cuda.to_device(np.zeros((n, 5), dtype=np.float64))
    rec_dist_d = cuda.to_device(np.zeros(n, dtype=np.float64))
    cdf_d = cuda.to_device(cdf.astype(np.float64))
    angles_d = cuda.to_device(angles_rad.astype(np.float64))
    rng_states = create_xoroshiro128p_states(n, seed=seed)

    blocks = (n + threads_per_block - 1) // threads_per_block

    t0 = time.perf_counter()
    kernel[blocks, threads_per_block](state_d, total_dist_d, rec_loc_d, rec_dist_d, cdf_d, angles_d, rng_states, n)
    cuda.synchronize()
    elapsed_s = time.perf_counter() - t0

    state = state_d.copy_to_host()
    rec_loc = rec_loc_d.copy_to_host()
    rec_dist = rec_dist_d.copy_to_host()

    det = state[:, 7] == 0.0
    n_packets = int(np.sum(det))
    return elapsed_s, rec_loc[det], rec_dist[det], state[det, 6], n_packets
