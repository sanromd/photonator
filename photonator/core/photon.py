"""PhotonBatch: vectorized photon state container."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Column indices into the N×8 state array
X, Y, Z, UX, UY, UZ, W, STATUS = range(8)

# Status codes
ACTIVE = 1
DETECTED = 0
TERMINATED = -1


class PhotonBatch:
    """Batch of N photons represented as an N×8 float64 array.

    Columns: x_m, y_m, z_m, ux, uy, uz, weight, status
    status: 1=active, 0=detected, -1=terminated
    """

    def __init__(self, n: int, rng: np.random.Generator | None = None) -> None:
        self.n = n
        self.rng = rng or np.random.default_rng()
        self._state: NDArray[np.float64] = np.zeros((n, 8), dtype=np.float64)
        self._state[:, W] = 1.0
        self._state[:, STATUS] = float(ACTIVE)
        self._path_length_m: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        self._rec_path_length_m: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        self._rec_loc: NDArray[np.float64] = np.zeros((n, 5), dtype=np.float64)

    # ------------------------------------------------------------------
    # Column accessors (views into _state — zero-copy)
    # ------------------------------------------------------------------

    @property
    def x_m(self) -> NDArray[np.float64]:
        return self._state[:, X]

    @property
    def y_m(self) -> NDArray[np.float64]:
        return self._state[:, Y]

    @property
    def z_m(self) -> NDArray[np.float64]:
        return self._state[:, Z]

    @property
    def ux(self) -> NDArray[np.float64]:
        return self._state[:, UX]

    @property
    def uy(self) -> NDArray[np.float64]:
        return self._state[:, UY]

    @property
    def uz(self) -> NDArray[np.float64]:
        return self._state[:, UZ]

    @property
    def weight(self) -> NDArray[np.float64]:
        return self._state[:, W]

    @property
    def status(self) -> NDArray[np.float64]:
        return self._state[:, STATUS]

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Boolean mask selecting photons still propagating."""
        return self._state[:, STATUS] == ACTIVE

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active_mask))

    @property
    def detected_mask(self) -> NDArray[np.bool_]:
        return self._state[:, STATUS] == DETECTED

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def state_view(self, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Return a view of the state rows selected by mask."""
        return self._state[mask]
