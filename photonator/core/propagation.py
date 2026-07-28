"""Vectorized CPU propagation loop (NumPy backend).

Faithfully translates mc_func_r6.m: Beer-Lambert path sampling, CDF
scattering, direction-cosine update, receiver-plane detection, and
roulette termination.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from photonator.constants import ROULETTE_CONST, ROULETTE_CONST_INV, min_weight_for_albedo
from photonator.core.photon import (
    ACTIVE,
    DETECTED,
    TERMINATED,
    PhotonBatch,
)
from photonator.core.scattering import update_direction

if TYPE_CHECKING:
    from photonator.core.receiver import Receiver
    from photonator.media.base import AbstractMedium
    from photonator.phase_functions.base import AbstractPhaseFunction


def propagate_cpu(
    batch: PhotonBatch,
    medium: AbstractMedium,
    phase_fn: AbstractPhaseFunction,
    receiver: Receiver,
    rx_plane_limits: bool = True,
    rx_x_lim_m: tuple[float, float] = (-3.0, 3.0),
    rx_y_lim_m: tuple[float, float] = (-3.0, 3.0),
) -> tuple[
    float,
    NDArray[np.float64],  # rec_loc (M, 5)
    NDArray[np.float64],  # distances_m (M,)
    NDArray[np.float64],  # rec_weights (M,)
    int,                  # total_rec_packets
]:
    """Run one batch of photons through the MC propagation loop.

    Parameters
    ----------
    batch : PhotonBatch, initialised by a beam (positions and directions set)
    medium : optical medium providing mu_a_per_m, mu_s_per_m, n
    phase_fn : phase function providing sample(n) → theta array
    receiver : Receiver instance (provides receiver_z_m)
    rx_plane_limits : if True, terminate photons that exit lateral bounds at rx plane
    rx_x_lim_m, rx_y_lim_m : lateral limits at receiver plane (m)

    Returns
    -------
    elapsed_s : wall-clock time (s)
    rec_loc : (M, 5) array [x, y, mu_x, mu_y, mu_z] at detection
    distances_m : path length at detection, shape (M,)
    rec_weights : photon weights at detection, shape (M,)
    total_rec_packets : number of detected photons
    """
    t0 = time.perf_counter()

    n = batch.n
    c_per_m = medium.mu_a_per_m + medium.mu_s_per_m
    a_per_m = medium.mu_a_per_m
    albedo = (c_per_m - a_per_m) / c_per_m if c_per_m > 0 else 0.0
    prob_survival = albedo
    inv_c = 1.0 / c_per_m
    min_w = min_weight_for_albedo(albedo)
    receiver_z_m = receiver.receiver_z_m
    rng = batch.rng

    rec_loc = np.zeros((n, 5), dtype=np.float64)
    rec_dist = np.zeros(n, dtype=np.float64)
    total_dist = np.zeros(n, dtype=np.float64)
    total_rec_packets = 0

    while batch.n_active > 0:
        active = batch.active_mask
        n_act = int(np.sum(active))
        idx = np.where(active)[0]

        # Sample path lengths (Beer-Lambert)
        r = -inv_c * np.log(rng.random(n_act))

        # Sample scattering angles and azimuth
        theta = phase_fn.sample(n_act, rng)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_act)

        ux_a = batch.ux[idx]
        uy_a = batch.uy[idx]
        uz_a = batch.uz[idx]
        x_a = batch.x_m[idx]
        y_a = batch.y_m[idx]
        z_a = batch.z_m[idx]

        x_step = r * ux_a
        y_step = r * uy_a
        z_step = r * uz_a

        new_z = z_a + z_step

        # --- Receiver-plane crossing ---
        crossed = new_z >= receiver_z_m

        if np.any(crossed):
            cidx = idx[crossed]
            uz_c = uz_a[crossed]
            # Guard against uz == 0 (should never happen physically)
            safe_uz = np.where(np.abs(uz_c) > 1e-15, uz_c, 1e-15)
            z_dist = receiver_z_m - z_a[crossed]
            dist_to_rec = z_dist / safe_uz

            x_rec = x_a[crossed] + z_dist * ux_a[crossed] / safe_uz
            y_rec = y_a[crossed] + z_dist * uy_a[crossed] / safe_uz

            # Lateral limits at receiver plane
            out_of_bounds = (
                rx_plane_limits
                and (
                    (x_rec < rx_x_lim_m[0])
                    | (x_rec > rx_x_lim_m[1])
                    | (y_rec < rx_y_lim_m[0])
                    | (y_rec > rx_y_lim_m[1])
                )
            )
            if isinstance(out_of_bounds, bool):
                out_of_bounds = np.full(len(cidx), out_of_bounds, dtype=bool)

            # Terminate out-of-bounds photons
            batch.status[cidx[out_of_bounds]] = float(TERMINATED)

            in_bounds = ~out_of_bounds
            rec_cidx = cidx[in_bounds]
            if len(rec_cidx) > 0:
                rec_loc[rec_cidx, 0] = x_rec[in_bounds]
                rec_loc[rec_cidx, 1] = y_rec[in_bounds]
                rec_loc[rec_cidx, 2] = ux_a[crossed][in_bounds]
                rec_loc[rec_cidx, 3] = uy_a[crossed][in_bounds]
                rec_loc[rec_cidx, 4] = uz_c[in_bounds]
                rec_dist[rec_cidx] = total_dist[rec_cidx] + dist_to_rec[in_bounds]
                total_dist[rec_cidx] += dist_to_rec[in_bounds]
                batch.status[rec_cidx] = float(DETECTED)
                total_rec_packets += len(rec_cidx)

        # --- Non-crossing photons ---
        no_cross = ~crossed
        if np.any(no_cross):
            ncidx = idx[no_cross]
            # Positions into the active-frame arrays (r, theta, phi, ux_a, ...).
            # Filtered in lockstep with ncidx so both stay aligned; indexing the
            # active-frame arrays with a mask derived from a *filtered* ncidx
            # would silently pair photons with other photons' angles/directions.
            sel = np.where(no_cross)[0]
            # Move photons
            batch.x_m[ncidx] = x_a[sel] + x_step[sel]
            batch.y_m[ncidx] = y_a[sel] + y_step[sel]
            batch.z_m[ncidx] = new_z[sel]
            total_dist[ncidx] += r[sel]

            # Terminate photons that backscattered past z=0
            below_zero = batch.z_m[ncidx] < 0.0
            batch.status[ncidx[below_zero]] = float(TERMINATED)

            still_active = ~below_zero
            ncidx = ncidx[still_active]
            sel = sel[still_active]
            if len(ncidx) == 0:
                continue

            # Implicit absorption (reduce weight by albedo each step)
            batch.weight[ncidx] *= prob_survival

            # Terminate immediately if weight hit zero (pure absorber / albedo=0)
            zero_weight = batch.weight[ncidx] == 0.0
            if np.any(zero_weight):
                batch.status[ncidx[zero_weight]] = float(TERMINATED)
                ncidx = ncidx[~zero_weight]
                sel = sel[~zero_weight]
            if len(ncidx) == 0:
                continue

            # Roulette termination
            low_w = batch.weight[ncidx] < min_w
            if np.any(low_w):
                roulette_rand = rng.random(int(np.sum(low_w)))
                low_idx = ncidx[low_w]
                survive = roulette_rand > ROULETTE_CONST_INV
                batch.status[low_idx[~survive]] = float(TERMINATED)
                batch.weight[low_idx[survive]] *= ROULETTE_CONST

            # Re-check still active after roulette
            still_act_mask = batch.status[ncidx] == ACTIVE
            ncidx = ncidx[still_act_mask]
            sel = sel[still_act_mask]
            if len(ncidx) == 0:
                continue

            # Update direction cosines
            ux_new, uy_new, uz_new = update_direction(
                ux_a[sel],
                uy_a[sel],
                uz_a[sel],
                theta[sel],
                phi[sel],
            )
            batch.ux[ncidx] = ux_new
            batch.uy[ncidx] = uy_new
            batch.uz[ncidx] = uz_new

    elapsed_s = time.perf_counter() - t0

    det = batch.detected_mask
    return (
        elapsed_s,
        rec_loc[det],
        rec_dist[det],
        batch.weight[det],
        total_rec_packets,
    )
