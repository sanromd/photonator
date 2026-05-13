"""scipy.io.loadmat wrapper for loading MATLAB .mat data files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as _sio


def load_mat_array(path: Path | str, variable: str) -> np.ndarray:
    """Load a 2-D array from a MATLAB .mat file.

    Parameters
    ----------
    path : path to the .mat file
    variable : name of the MATLAB variable to load

    Returns
    -------
    numpy array (dtype float64)
    """
    mat = _sio.loadmat(str(path))
    data = mat[variable]
    return np.asarray(data, dtype=np.float64)


def load_mat_variable(path: Path | str, variable: str) -> np.ndarray:
    """Load an arbitrary variable from a MATLAB .mat file.

    Parameters
    ----------
    path : path to the .mat file
    variable : name of the MATLAB variable

    Returns
    -------
    numpy array
    """
    mat = _sio.loadmat(str(path))
    return np.asarray(mat[variable])
