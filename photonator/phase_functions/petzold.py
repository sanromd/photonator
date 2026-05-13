"""Petzold measured ocean volume scattering functions.

Loads petzold_ocean.mat (columns: angle_rad, harbor VSF, coastal VSF, clear VSF)
and builds a CDF for the chosen water condition.  Matches generate_scatter.m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonator.phase_functions.base import AbstractPhaseFunction
from photonator.io.mat_loader import load_mat_array

# Path to the .mat data file relative to this file's package root
_DATA_DIR = Path(__file__).parent.parent.parent  # repo root


class PetzoldPhaseFunction(AbstractPhaseFunction):
    """Phase function built from Petzold's measured ocean VSF.

    Parameters
    ----------
    water_type : one of "harbor", "coastal", "clear"
    mat_path : path to petzold_ocean.mat (defaults to repo root)
    """

    _COL_MAP = {"harbor": 1, "coastal": 2, "clear": 3}

    def __init__(
        self,
        water_type: str = "clear",
        mat_path: Path | str | None = None,
    ) -> None:
        if water_type not in self._COL_MAP:
            raise ValueError(f"water_type must be one of {list(self._COL_MAP)}")
        self.water_type = water_type

        mat_path = Path(mat_path) if mat_path else _DATA_DIR / "petzold_ocean.mat"
        data = load_mat_array(mat_path, "petzold_ocean")  # (K, 4)

        angles_rad: NDArray[np.float64] = data[:, 0].astype(np.float64)
        vsf: NDArray[np.float64] = data[:, self._COL_MAP[water_type]].astype(np.float64)

        self._cdf, self._angles_rad = self.build_cdf(angles_rad, vsf)

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Sample n polar angles from the Petzold VSF CDF."""
        return self.sample_from_cdf(self._cdf, self._angles_rad, n, rng)
