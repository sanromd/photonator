"""Convert legacy MATLAB .mat data files to HDF5.

Run from the repo root:
    python scripts/convert_mat_to_hdf5.py

Each .mat file at the repo root is converted to an equivalent .h5 file
under data/, preserving all variable names as top-level HDF5 datasets.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MAT_FILES = [
    "petzold_ocean.mat",
    "petzold_data_maalox_orig.mat",
    "maalox_alan_orig.mat",
    "widemann_maalox.mat",
]


def convert(mat_path: Path, h5_path: Path) -> None:
    mat = sio.loadmat(str(mat_path))
    variables = {k: v for k, v in mat.items() if not k.startswith("__")}
    with h5py.File(h5_path, "w") as f:
        for name, array in variables.items():
            f.create_dataset(name, data=np.asarray(array, dtype=np.float64))
    print(f"  {mat_path.name} -> {h5_path.name}  {list(variables)}")


def main() -> None:
    print(f"Writing HDF5 files to {DATA_DIR}/")
    for mat_name in MAT_FILES:
        mat_path = REPO_ROOT / mat_name
        if not mat_path.exists():
            print(f"  SKIP {mat_name} (not found)")
            continue
        h5_name = mat_path.stem + ".h5"
        convert(mat_path, DATA_DIR / h5_name)
    print("Done.")


if __name__ == "__main__":
    main()
