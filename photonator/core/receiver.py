"""Receiver: detects photons at the receiver plane with Fresnel and FOV filtering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from photonator.constants import N_AIR, N_WATER, CRIT_ANG_COS_WATER_AIR


# Two-term Gaussian FOV weighting from mc_rec_r5.m
_A1, _B1, _C1 = 0.7985, 0.0187, 0.03437
_A2, _B2, _C2 = 0.7121, -0.02337, 0.03117


@dataclass
class ReceivedPhotons:
    """Statistics and raw data from detected photons."""

    count: int = 0
    power: float = 0.0
    angle_mean_rad: float = 0.0      # mean of uz (cos of arrival angle)
    angle_var: float = 0.0
    dist_mean_m: float = 0.0
    dist_var: float = 0.0
    weight_mean: float = 0.0
    weight_var: float = 0.0
    reflected: int = 0               # photons beyond critical angle
    # Raw arrays (pre-allocated, cropped on finalise)
    _angles: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    _distances_m: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    _weights: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))


class Receiver:
    """Single circular receiver at the z = receiver_z_m plane.

    Applies:
    - Total-internal-reflection cutoff (water→air)
    - Unpolarised Fresnel transmission
    - Circular aperture check
    - FOV check (half-angle)
    - Two-term Gaussian FOV weighting (from mc_rec_r5.m)
    - Welford online mean/variance

    Parameters
    ----------
    receiver_z_m : axial position of the receiver plane (m)
    pos_x_m, pos_y_m : receiver centre coordinates (m)
    aperture_m : diameter of receiver aperture (m)
    fov_rad : full field-of-view angle (rad)
    n_water : refractive index of the propagation medium
    n_air : refractive index on exit side
    """

    def __init__(
        self,
        receiver_z_m: float,
        pos_x_m: float = 0.0,
        pos_y_m: float = 0.0,
        aperture_m: float = 0.8,
        fov_rad: float = float(np.pi / 2),
        n_water: float = N_WATER,
        n_air: float = N_AIR,
    ) -> None:
        self.receiver_z_m = receiver_z_m
        self.pos_x_m = pos_x_m
        self.pos_y_m = pos_y_m
        self.radius_m = aperture_m / 2.0
        self.cos_fov_half = float(np.cos(fov_rad / 2.0))
        self.n_water = n_water
        self.n_air = n_air
        self._crit_ang_cos = float(np.sqrt(1.0 - (n_air / n_water) ** 2))
        # Welford accumulators
        self._count = 0
        self._power = 0.0
        self._angle_mean = 0.0
        self._angle_M2 = 0.0
        self._dist_mean = 0.0
        self._dist_M2 = 0.0
        self._weight_mean = 0.0
        self._weight_M2 = 0.0
        self._reflected = 0

    def detect(
        self,
        rec_loc: NDArray[np.float64],   # (M, 5): x, y, mu_x, mu_y, mu_z
        distances_m: NDArray[np.float64],  # (M,)
        weights: NDArray[np.float64],      # (M,)
        n_tx_photons: int,
    ) -> ReceivedPhotons:
        """Process photons that reached the receiver plane.

        Applies Fresnel, critical angle, aperture, FOV, and Welford stats.
        Matches mc_rec_r5.m logic exactly.
        """
        ph_x = rec_loc[:, 0]
        ph_y = rec_loc[:, 1]
        mu_z = rec_loc[:, 4]
        w = weights.copy()

        # Critical angle filter
        beyond_critical = mu_z <= self._crit_ang_cos
        self._reflected += int(np.sum(beyond_critical))
        keep = ~beyond_critical

        ph_x = ph_x[keep]
        ph_y = ph_y[keep]
        mu_z_k = mu_z[keep]
        w_k = w[keep]
        dist_k = distances_m[keep]

        # Fresnel transmission
        cos_exit = np.sqrt(np.maximum(1.0 - (self.n_water / self.n_air) ** 2 * (1.0 - mu_z_k**2), 0.0))
        rp = (mu_z_k - self.n_water * cos_exit) / (mu_z_k + self.n_water * cos_exit)
        rs = (cos_exit - self.n_water * mu_z_k) / (cos_exit + self.n_water * mu_z_k)
        R = (rp**2 + rs**2) / 2.0
        T = 1.0 - R
        w_k = w_k * T

        # Aperture check
        dist_from_centre = np.sqrt((ph_x - self.pos_x_m) ** 2 + (ph_y - self.pos_y_m) ** 2)
        in_aperture = dist_from_centre <= self.radius_m

        # FOV check
        in_fov = mu_z_k >= self.cos_fov_half

        valid = in_aperture & in_fov
        mu_z_v = mu_z_k[valid]
        w_v = w_k[valid]
        dist_v = dist_k[valid]

        # FOV Gaussian weighting
        arr_ang = np.arccos(np.clip(mu_z_v, -1.0, 1.0))
        fov_w = (
            _A1 * np.exp(-(((arr_ang - _B1) / _C1) ** 2))
            + _A2 * np.exp(-(((arr_ang - _B2) / _C2) ** 2))
        )
        w_final = w_v * fov_w

        # Welford online mean/variance for angle, distance, weight
        for i in range(len(w_final)):
            self._count += 1
            n = self._count
            ang = float(mu_z_v[i])
            dist = float(dist_v[i])
            wt = float(w_final[i])

            d_ang = ang - self._angle_mean
            self._angle_mean += d_ang / n
            self._angle_M2 += d_ang * (ang - self._angle_mean)

            d_dist = dist - self._dist_mean
            self._dist_mean += d_dist / n
            self._dist_M2 += d_dist * (dist - self._dist_mean)

            d_wt = wt - self._weight_mean
            self._weight_mean += d_wt / n
            self._weight_M2 += d_wt * (wt - self._weight_mean)

        self._power += float(np.sum(w_final))

        result = ReceivedPhotons(
            count=self._count,
            power=self._power,
            angle_mean_rad=self._angle_mean,
            angle_var=self._angle_M2 / max(self._count - 1, 1),
            dist_mean_m=self._dist_mean,
            dist_var=self._dist_M2 / max(self._count - 1, 1),
            weight_mean=self._count * self._weight_mean / max(n_tx_photons, 1),
            weight_var=self._weight_M2 / max(self._count - 1, 1),
            reflected=self._reflected,
        )
        return result

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._count = 0
        self._power = 0.0
        self._angle_mean = 0.0
        self._angle_M2 = 0.0
        self._dist_mean = 0.0
        self._dist_M2 = 0.0
        self._weight_mean = 0.0
        self._weight_M2 = 0.0
        self._reflected = 0
