"""Vectorized CPU propagation through an axially layered medium.

Extends the homogeneous loop in :mod:`photonator.core.propagation` with
per-layer optical properties and geometric-optics interface physics:

- Free path is sampled with the *local* layer's attenuation coefficient.
  When the sampled path would cross a layer boundary, the photon is moved
  to the boundary and the step is re-drawn there — unbiased because the
  exponential distribution is memoryless.
- At each interface the photon is specularly reflected with the
  unpolarised Fresnel probability (total internal reflection included)
  or refracted per Snell's law (transverse direction cosines scale by
  n1/n2, which preserves the unit norm exactly).
- Scattering angles are drawn from the layer's own phase function when
  ``Layer.phase_fn`` is set, else from the global one.

Photons crossing the bottom of layer 0 (z=0) terminate, matching the
homogeneous loop's z<0 termination.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from photonator.constants import ROULETTE_CONST, ROULETTE_CONST_INV, min_weight_for_albedo
from photonator.core.photon import ACTIVE, DETECTED, TERMINATED, PhotonBatch
from photonator.core.scattering import update_direction

if TYPE_CHECKING:
    from photonator.core.receiver import Receiver
    from photonator.media.layers import LayeredMedium
    from photonator.phase_functions.base import AbstractPhaseFunction

# Nudge distance across a boundary so the photon's next layer lookup
# lands on the correct side of the interface (metres).
_BOUNDARY_EPS_M: float = 1e-9


def propagate_layered_cpu(
    batch: PhotonBatch,
    medium: LayeredMedium,
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
    """Run one batch of photons through an axially layered medium.

    Parameters
    ----------
    batch : PhotonBatch, initialised by a beam
    medium : LayeredMedium whose layers cover [0, receiver_z_m]
    phase_fn : global phase function; overridden per layer by Layer.phase_fn
    receiver : Receiver instance (provides receiver_z_m)
    rx_plane_limits : if True, terminate photons outside lateral bounds at rx plane
    rx_x_lim_m, rx_y_lim_m : lateral limits at receiver plane (m)

    Returns
    -------
    Same tuple as :func:`photonator.core.propagation.propagate_cpu`.
    """
    t0 = time.perf_counter()

    layers = medium.layers
    n_layers = len(layers)
    z_starts = np.array([lay.z_start_m for lay in layers], dtype=np.float64)
    z_ends = np.array([lay.z_end_m for lay in layers], dtype=np.float64)
    mu_a_l = np.array([lay.medium.mu_a_per_m for lay in layers], dtype=np.float64)
    mu_s_l = np.array([lay.medium.mu_s_per_m for lay in layers], dtype=np.float64)
    c_l = mu_a_l + mu_s_l
    albedo_l = np.where(c_l > 0, mu_s_l / np.maximum(c_l, 1e-30), 0.0)
    n_l = np.array([lay.medium.n for lay in layers], dtype=np.float64)
    min_w_l = np.array([min_weight_for_albedo(a) for a in albedo_l], dtype=np.float64)
    layer_pfs = [getattr(lay, "phase_fn", None) or phase_fn for lay in layers]

    n = batch.n
    receiver_z_m = receiver.receiver_z_m
    rng = batch.rng

    rec_loc = np.zeros((n, 5), dtype=np.float64)
    rec_dist = np.zeros(n, dtype=np.float64)
    total_dist = batch._path_length_m
    total_rec_packets = 0

    while batch.n_active > 0:
        idx = np.where(batch.active_mask)[0]
        n_act = len(idx)

        x_a = batch.x_m[idx]
        y_a = batch.y_m[idx]
        z_a = batch.z_m[idx]
        ux_a = batch.ux[idx]
        uy_a = batch.uy[idx]
        uz_a = batch.uz[idx]

        # Layer index per photon
        li = np.clip(np.searchsorted(z_starts, z_a, side="right") - 1, 0, n_layers - 1)

        # Free path with the local attenuation coefficient
        r = -np.log(np.maximum(rng.random(n_act), 1e-300)) / np.maximum(c_l[li], 1e-30)

        # Scattering angles: per-layer phase functions
        theta = np.empty(n_act, dtype=np.float64)
        for k in np.unique(li):
            in_k = li == k
            theta[in_k] = layer_pfs[k].sample(int(np.sum(in_k)), rng)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_act)

        # Distance to the next layer boundary along the current direction
        going_up = uz_a > 0.0
        z_bound = np.where(going_up, z_ends[li], z_starts[li])
        safe_uz = np.where(np.abs(uz_a) > 1e-15, uz_a, 1e-15)
        dist_bound = np.where(np.abs(uz_a) > 1e-15, (z_bound - z_a) / safe_uz, np.inf)
        # Top layer going up: no interface above (free run toward the receiver)
        dist_bound = np.where(going_up & (li == n_layers - 1), np.inf, dist_bound)

        # Distance to the receiver plane (up-going photons only)
        dist_rec = np.where(uz_a > 1e-15, (receiver_z_m - z_a) / safe_uz, np.inf)

        # Event partition: detection, interface crossing, or scattering
        detect_e = dist_rec <= np.minimum(r, dist_bound)
        iface_e = ~detect_e & (dist_bound < r)
        scatter_e = ~(detect_e | iface_e)

        # --- Detection at the receiver plane ---
        if np.any(detect_e):
            sd = np.where(detect_e)[0]
            cidx = idx[sd]
            t_rec = dist_rec[sd]
            x_rec = x_a[sd] + t_rec * ux_a[sd]
            y_rec = y_a[sd] + t_rec * uy_a[sd]

            out_of_bounds = rx_plane_limits & (
                (x_rec < rx_x_lim_m[0])
                | (x_rec > rx_x_lim_m[1])
                | (y_rec < rx_y_lim_m[0])
                | (y_rec > rx_y_lim_m[1])
            )
            if isinstance(out_of_bounds, (bool, np.bool_)):
                out_of_bounds = np.full(len(cidx), bool(out_of_bounds), dtype=bool)

            batch.status[cidx[out_of_bounds]] = float(TERMINATED)

            in_b = ~out_of_bounds
            rec_cidx = cidx[in_b]
            if len(rec_cidx) > 0:
                rec_loc[rec_cidx, 0] = x_rec[in_b]
                rec_loc[rec_cidx, 1] = y_rec[in_b]
                rec_loc[rec_cidx, 2] = ux_a[sd][in_b]
                rec_loc[rec_cidx, 3] = uy_a[sd][in_b]
                rec_loc[rec_cidx, 4] = uz_a[sd][in_b]
                total_dist[rec_cidx] += t_rec[in_b]
                rec_dist[rec_cidx] = total_dist[rec_cidx]
                batch.status[rec_cidx] = float(DETECTED)
                total_rec_packets += len(rec_cidx)

        # --- Interface crossing: move to boundary, reflect or refract ---
        if np.any(iface_e):
            si = np.where(iface_e)[0]
            iidx = idx[si]
            t_b = dist_bound[si]
            batch.x_m[iidx] = x_a[si] + t_b * ux_a[si]
            batch.y_m[iidx] = y_a[si] + t_b * uy_a[si]
            total_dist[iidx] += t_b

            zb = z_bound[si]
            uzi = uz_a[si]
            uxi = ux_a[si]
            uyi = uy_a[si]
            lii = li[si]
            upi = uzi > 0.0

            # Crossing the bottom of layer 0 == exiting z<0: terminate
            exits = ~upi & (lii == 0)
            batch.z_m[iidx] = zb
            batch.status[iidx[exits]] = float(TERMINATED)

            keep = ~exits
            iidx = iidx[keep]
            if len(iidx) > 0:
                zb = zb[keep]
                uzi = uzi[keep]
                uxi = uxi[keep]
                uyi = uyi[keep]
                lii = lii[keep]
                upi = upi[keep]

                li_next = np.where(upi, lii + 1, lii - 1)
                n1 = n_l[lii]
                n2 = n_l[li_next]
                ratio = n1 / n2

                sin2_t = ratio**2 * (1.0 - uzi**2)
                tir = sin2_t > 1.0
                cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
                cos_i = np.abs(uzi)

                # Unpolarised Fresnel reflectance (R=1 under TIR)
                rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
                rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
                refl_prob = np.where(tir, 1.0, (rs**2 + rp**2) / 2.0)

                reflect = rng.random(len(iidx)) < refl_prob
                sgn = np.sign(uzi)
                batch.uz[iidx] = np.where(reflect, -uzi, sgn * cos_t)
                batch.ux[iidx] = np.where(reflect, uxi, uxi * ratio)
                batch.uy[iidx] = np.where(reflect, uyi, uyi * ratio)
                # Nudge off the boundary: back into the current layer on
                # reflection, into the next layer on refraction.
                batch.z_m[iidx] = np.where(
                    reflect, zb - sgn * _BOUNDARY_EPS_M, zb + sgn * _BOUNDARY_EPS_M
                )

        # --- Scattering interaction inside the current layer ---
        if np.any(scatter_e):
            sel = np.where(scatter_e)[0]
            ncidx = idx[sel]

            batch.x_m[ncidx] = x_a[sel] + r[sel] * ux_a[sel]
            batch.y_m[ncidx] = y_a[sel] + r[sel] * uy_a[sel]
            batch.z_m[ncidx] = z_a[sel] + r[sel] * uz_a[sel]
            total_dist[ncidx] += r[sel]

            # Implicit absorption with the local layer's albedo
            batch.weight[ncidx] *= albedo_l[li[sel]]

            zero_weight = batch.weight[ncidx] == 0.0
            if np.any(zero_weight):
                batch.status[ncidx[zero_weight]] = float(TERMINATED)
                ncidx = ncidx[~zero_weight]
                sel = sel[~zero_weight]
            if len(ncidx) == 0:
                continue

            low_w = batch.weight[ncidx] < min_w_l[li[sel]]
            if np.any(low_w):
                roulette_rand = rng.random(int(np.sum(low_w)))
                low_idx = ncidx[low_w]
                survive = roulette_rand > ROULETTE_CONST_INV
                batch.status[low_idx[~survive]] = float(TERMINATED)
                batch.weight[low_idx[survive]] *= ROULETTE_CONST

            still_act = batch.status[ncidx] == ACTIVE
            ncidx = ncidx[still_act]
            sel = sel[still_act]
            if len(ncidx) == 0:
                continue

            ux_new, uy_new, uz_new = update_direction(
                ux_a[sel], uy_a[sel], uz_a[sel], theta[sel], phi[sel]
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
