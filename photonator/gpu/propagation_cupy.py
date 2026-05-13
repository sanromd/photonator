"""CuPy vectorized GPU propagation — drop-in replacement for NumPy backend.

Replaces np.* calls with cp.* for GPU-resident arrays.  The logic is
identical to core/propagation.py; refer to that module for algorithmic
documentation.

Requires: cupy (install with pip install cupy-cuda12x or similar).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from photonator.core.photon import PhotonBatch
    from photonator.phase_functions.base import AbstractPhaseFunction
    from photonator.media.base import AbstractMedium
    from photonator.core.receiver import Receiver

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def propagate_cupy(
    batch: "PhotonBatch",
    medium: "AbstractMedium",
    phase_fn: "AbstractPhaseFunction",
    receiver: "Receiver",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run MC propagation on GPU using CuPy vectorized arrays.

    Falls back to CPU (core.propagation.propagate_cpu) if CuPy is unavailable.

    Parameters
    ----------
    batch : PhotonBatch (initialised by beam)
    medium : optical medium
    phase_fn : phase function (sample method must return CPU numpy arrays)
    receiver : Receiver instance

    Returns
    -------
    elapsed_s, rec_loc (M,5), distances_m (M,), rec_weights (M,), n_packets
    """
    if not _CUPY_AVAILABLE:
        from photonator.core.propagation import propagate_cpu
        return propagate_cpu(batch, medium, phase_fn, receiver)

    from photonator.constants import ROULETTE_CONST_INV, min_weight_for_albedo
    from photonator.core.photon import ACTIVE, DETECTED, TERMINATED

    n = batch.n
    c_per_m = medium.mu_a_per_m + medium.mu_s_per_m
    albedo = medium.mu_s_per_m / c_per_m if c_per_m > 0 else 0.0
    inv_c = 1.0 / c_per_m
    min_w = min_weight_for_albedo(albedo)
    receiver_z = receiver.receiver_z_m
    rng = batch.rng
    MAX_UZ = 1.0 - 1e-12

    # Upload state to GPU
    state = cp.asarray(batch._state, dtype=cp.float64)
    total_dist = cp.zeros(n, dtype=cp.float64)
    rec_loc = cp.zeros((n, 5), dtype=cp.float64)
    rec_dist = cp.zeros(n, dtype=cp.float64)

    X, Y, Z, UX, UY, UZ, W, STATUS = range(8)

    t0 = time.perf_counter()

    while True:
        active = state[:, STATUS] == ACTIVE
        if not bool(cp.any(active)):
            break
        idx = cp.where(active)[0]
        n_act = int(idx.shape[0])

        # Sample path lengths and angles on CPU, upload
        r_cpu = -inv_c * np.log(np.maximum(rng.random(n_act), 1e-15))
        theta_cpu = phase_fn.sample(n_act, rng)
        phi_cpu = rng.uniform(0.0, 2.0 * np.pi, n_act)

        r = cp.asarray(r_cpu)
        theta = cp.asarray(theta_cpu)
        phi = cp.asarray(phi_cpu)

        ux = state[idx, UX]
        uy = state[idx, UY]
        uz = state[idx, UZ]

        new_z = state[idx, Z] + r * uz
        crossed = new_z >= receiver_z

        # Receiver crossings
        if bool(cp.any(crossed)):
            cidx = idx[crossed]
            uz_c = uz[crossed]
            safe_uz = cp.where(cp.abs(uz_c) > 1e-15, uz_c, 1e-15)
            z_dist = receiver_z - state[cidx, Z]
            dist_to_rec = z_dist / safe_uz
            rec_loc[cidx, 0] = state[cidx, X] + z_dist * ux[crossed] / safe_uz
            rec_loc[cidx, 1] = state[cidx, Y] + z_dist * uy[crossed] / safe_uz
            rec_loc[cidx, 2] = ux[crossed]
            rec_loc[cidx, 3] = uy[crossed]
            rec_loc[cidx, 4] = uz_c
            rec_dist[cidx] = total_dist[cidx] + dist_to_rec
            total_dist[cidx] += dist_to_rec
            state[cidx, STATUS] = DETECTED

        # Non-crossing photons
        no_cross = ~crossed
        if bool(cp.any(no_cross)):
            ncidx = idx[no_cross]
            state[ncidx, X] += r[no_cross] * ux[no_cross]
            state[ncidx, Y] += r[no_cross] * uy[no_cross]
            state[ncidx, Z] = new_z[no_cross]
            total_dist[ncidx] += r[no_cross]

            below = state[ncidx, Z] < 0.0
            state[ncidx[below], STATUS] = TERMINATED

            alive = ~below
            ncidx = ncidx[alive]
            if not int(ncidx.shape[0]):
                continue

            state[ncidx, W] *= albedo

            low_w = state[ncidx, W] < min_w
            if bool(cp.any(low_w)):
                low_idx = ncidx[low_w]
                n_low = int(low_idx.shape[0])
                u_rou = cp.asarray(rng.random(n_low))
                survive = u_rou > ROULETTE_CONST_INV
                state[low_idx[~survive], STATUS] = TERMINATED
                state[low_idx[survive], W] *= 10.0

            still = state[ncidx, STATUS] == ACTIVE
            ncidx = ncidx[still]
            if not int(ncidx.shape[0]):
                continue

            # Remap indices for angle arrays
            nc_in_alive = cp.where(still)[0]
            t = theta[no_cross][alive][nc_in_alive]
            p = phi[no_cross][alive][nc_in_alive]
            sin_t = cp.sin(t)
            cos_t = cp.cos(t)
            cos_p = cp.cos(p)
            sin_p = cp.sin(p)

            old_ux = state[ncidx, UX]
            old_uy = state[ncidx, UY]
            old_uz = state[ncidx, UZ]
            near_axis = cp.abs(old_uz) > MAX_UZ
            sqrt_1_uz2 = cp.sqrt(cp.maximum(1.0 - old_uz**2, 0.0))

            ux_new = cp.where(
                near_axis, sin_t * cos_p,
                (sin_t / cp.where(sqrt_1_uz2 > 1e-15, sqrt_1_uz2, 1.0)) * (old_ux * old_uz * cos_p - old_uy * sin_p) + old_ux * cos_t,
            )
            uy_new = cp.where(
                near_axis, sin_t * sin_p,
                (sin_t / cp.where(sqrt_1_uz2 > 1e-15, sqrt_1_uz2, 1.0)) * (old_uy * old_uz * cos_p + old_ux * sin_p) + old_uy * cos_t,
            )
            uz_new = cp.where(
                near_axis, cp.sign(old_uz) * cos_t,
                -sin_t * cos_p * sqrt_1_uz2 + old_uz * cos_t,
            )
            norm = cp.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
            needs_norm = cp.abs(1.0 - norm) > 1e-11
            ux_new = cp.where(needs_norm, ux_new / norm, ux_new)
            uy_new = cp.where(needs_norm, uy_new / norm, uy_new)
            uz_new = cp.where(needs_norm, uz_new / norm, uz_new)
            state[ncidx, UX] = ux_new
            state[ncidx, UY] = uy_new
            state[ncidx, UZ] = uz_new

    elapsed_s = time.perf_counter() - t0

    state_cpu = cp.asnumpy(state)
    rec_loc_cpu = cp.asnumpy(rec_loc)
    rec_dist_cpu = cp.asnumpy(rec_dist)

    det = state_cpu[:, STATUS] == DETECTED
    n_packets = int(np.sum(det))
    return elapsed_s, rec_loc_cpu[det], rec_dist_cpu[det], state_cpu[det, W], n_packets
