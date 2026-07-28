"""HDF5 save/load for SimulationResult."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from photonator.simulation import SimulationResult


def save_hdf5(result: SimulationResult, path: Path | str) -> None:
    """Save a SimulationResult to an HDF5 file.

    File structure::

        result.h5
        ├── config/          # simulation parameters as attributes
        ├── receiver/        # power, count, angle_mean, angle_var, ...
        ├── photons/         # raw photon-plane data (optional)
        └── metadata/        # timestamp, git hash, version

    Parameters
    ----------
    result : SimulationResult from Simulation.run()
    path : output file path (.h5 or .hdf5)
    """
    import datetime

    import h5py

    import photonator

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Receiver statistics
        rx = f.create_group("receiver")
        rx.attrs["power"] = result.total_power
        rx.attrs["count"] = result.total_packets
        rx.attrs["angle_mean_rad"] = result.angle_mean_rad
        rx.attrs["angle_var"] = result.angle_var
        rx.attrs["dist_mean_m"] = result.dist_mean_m
        rx.attrs["dist_var"] = result.dist_var
        rx.attrs["weight_mean"] = result.weight_mean
        rx.attrs["weight_var"] = result.weight_var
        rx.attrs["reflected"] = result.reflected

        # Raw photon data
        ph = f.create_group("photons")
        if result.rec_loc is not None:
            ph.create_dataset("rec_loc", data=result.rec_loc, compression="gzip")
        if result.distances_m is not None:
            ph.create_dataset("distances_m", data=result.distances_m, compression="gzip")
        if result.rec_weights is not None:
            ph.create_dataset("weights", data=result.rec_weights, compression="gzip")

        # Config
        cfg = f.create_group("config")
        cfg.attrs["n_photons"] = result.n_photons
        cfg.attrs["n_batches"] = result.n_batches
        cfg.attrs["elapsed_s"] = result.elapsed_s

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta.attrs["photonator_version"] = photonator.__version__
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = "unknown"
        meta.attrs["git_hash"] = git_hash


def load_hdf5(path: Path | str) -> dict:
    """Load a SimulationResult HDF5 file and return a plain dict of values."""
    import h5py

    result: dict = {}
    with h5py.File(str(path), "r") as f:
        for group_name in f:
            grp = f[group_name]
            result[group_name] = {}
            for k, v in grp.attrs.items():
                result[group_name][k] = v
            for k in grp:
                result[group_name][k] = np.array(grp[k])
    return result
