"""Vectorized direction-cosine update after a scattering event."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MAX_UZ: float = 1.0 - 1e-12


def update_direction(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    uz: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Rotate direction cosines (ux, uy, uz) by polar θ and azimuthal φ.

    Faithfully vectorises the rotation formula in mc_func_r6.m.
    When |uz| ≈ 1 the degenerate-case formula is used element-wise.

    Parameters
    ----------
    ux, uy, uz : float64 arrays of shape (N,) — current direction cosines
    theta : polar scattering angle (rad), shape (N,)
    phi   : azimuthal angle (rad), shape (N,)

    Returns
    -------
    ux_new, uy_new, uz_new : updated, normalised direction cosines
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    near_axis = np.abs(uz) > _MAX_UZ

    sqrt_1_uz2 = np.sqrt(np.maximum(1.0 - uz**2, 0.0))
    # Protect against division by zero when uz ≈ ±1 (near-axis photons).
    # np.where evaluates both branches, so we must guard the denominator even
    # for elements where near_axis=True.
    safe_sqrt = np.where(sqrt_1_uz2 > 1e-15, sqrt_1_uz2, 1.0)

    # General case
    ux_new = np.where(
        near_axis,
        sin_t * cos_p,
        (sin_t / safe_sqrt) * (ux * uz * cos_p - uy * sin_p) + ux * cos_t,
    )
    uy_new = np.where(
        near_axis,
        sin_t * sin_p,
        (sin_t / safe_sqrt) * (uy * uz * cos_p + ux * sin_p) + uy * cos_t,
    )
    uz_new = np.where(
        near_axis,
        np.sign(uz) * cos_t,
        -sin_t * cos_p * sqrt_1_uz2 + uz * cos_t,
    )

    # Re-normalise (guards floating-point drift)
    norm = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    needs_norm = np.abs(1.0 - norm) > 1e-11
    ux_new = np.where(needs_norm, ux_new / norm, ux_new)
    uy_new = np.where(needs_norm, uy_new / norm, uy_new)
    uz_new = np.where(needs_norm, uz_new / norm, uz_new)

    return ux_new, uy_new, uz_new
