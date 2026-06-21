"""Loaders for legacy MATLAB .mat files and HDF5 data files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as _sio


def load_hdf5_array(path: Path | str, variable: str) -> np.ndarray:
    """Load a dataset from an HDF5 file.

    Parameters
    ----------
    path : path to the .h5 file
    variable : HDF5 dataset name

    Returns
    -------
    numpy array (dtype float64)
    """
    import h5py
    with h5py.File(str(path), "r") as f:
        return np.array(f[variable], dtype=np.float64)


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
