"""Mie phase function from a tabulated VSF file.

Loads a user-supplied .mat or HDF5 file containing angle_rad and mie_vsf
columns and builds a CDF for sampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonator.phase_functions.base import AbstractPhaseFunction


class MiePhaseFunction(AbstractPhaseFunction):
    """Phase function built from a tabulated Mie VSF.

    Parameters
    ----------
    path : path to a .mat file containing variables ``angle_rad`` and ``mie_vsf``,
           or an HDF5 file with datasets of the same names.
    angle_key : key for the angle array in the file
    vsf_key : key for the VSF array in the file
    """

    def __init__(
        self,
        path: Path | str,
        angle_key: str = "angle_rad",
        vsf_key: str = "mie_vsf",
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".mat":
            from photonator.io.mat_loader import load_mat_variable
            angles_rad = load_mat_variable(path, angle_key).ravel().astype(np.float64)
            vsf = load_mat_variable(path, vsf_key).ravel().astype(np.float64)
        elif path.suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as f:
                angles_rad = np.array(f[angle_key], dtype=np.float64).ravel()
                vsf = np.array(f[vsf_key], dtype=np.float64).ravel()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._cdf, self._angles_rad = self.build_cdf(angles_rad, vsf)

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Sample n polar angles from the Mie VSF CDF."""
        return self.sample_from_cdf(self._cdf, self._angles_rad, n, rng)
