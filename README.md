# Photonator

Monte Carlo photon-transport simulator for scattering and absorbing liquids —
seawater, brines, oils, and mixtures. Models how beam type, wavelength, geometry,
and material composition affect optical propagation through random media.

Applications: underwater optical communications, ocean remote sensing, random-media research.

---

## Repository layout

```
photonator/       Python package — core engine, beams, media, phase functions, GPU backends, I/O
data/             VSF lookup tables (HDF5): Petzold ocean, Maalox suspensions
docs/             Reference documentation (physics, architecture, validation, GPU setup)
tests/            pytest suite (38 tests)
scripts/          Utility scripts (data conversion, benchmarks)
environment.yml   micromamba/conda environment
pyproject.toml    Package metadata and build config
```

---

## Quick start

### 1. Create the environment

```bash
micromamba env create -f environment.yml
micromamba activate photonator
```

This installs Python 3.11, NumPy, SciPy, h5py, and the `photonator` package in editable mode.

For GPU support (NVIDIA CUDA), uncomment the GPU block in `environment.yml` and run:

```bash
micromamba env update -f environment.yml
```

### 2. Run a simulation

```python
from photonator import Simulation
from photonator.beam import GaussianBeam
from photonator.media import Water
from photonator.phase_functions import PetzoldPhaseFunction
from photonator.core.receiver import Receiver

sim = Simulation(
    medium=Water(turbidity=1.0),
    beam=GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.00075),
    phase_fn=PetzoldPhaseFunction(water_type="clear"),
    receiver=Receiver(receiver_z_m=16.0, aperture_m=0.8, fov_rad=1.5708),
    n_photons=100_000,
    n_batches=10,
)
result = sim.run()
```

### 3. Save results

```python
from photonator.io import save_hdf5
save_hdf5(result, "output.h5")
```

### 4. Run tests

```bash
pytest tests/ -q
```

---

## Documentation

| Document | Contents |
|---|---|
| `docs/physics.md` | Beer-Lambert, HG phase function, Fresnel, Snell, beam profiles |
| `docs/architecture.md` | Class hierarchy, data flow, extension points |
| `docs/mcns.md` | Monte Carlo numerics: path sampling, CDF, roulette, Welford stats |
| `docs/materials.md` | Optical property sources and override format |
| `docs/validation.md` | Pass criteria and test commands |
| `docs/gpu_setup.md` | CUDA / Numba / CuPy installation guide |
| `docs/summary.md` | Original MATLAB codebase map and MATLAB-to-Python translation table |

---

## Historical MATLAB codebase

The original simulator (2009–2011, William Cox) was written in MATLAB. The Python
rewrite replaces the per-photon serial loop with vectorized NumPy/CuPy operations
and adds GPU support, pluggable components, and HDF5 I/O.

The MATLAB source files are preserved for reference:

```bash
# Browse the archive branch
git checkout archive/matlab-original

# Or check out the permanent tag
git checkout matlab-archive
```

For a full description of every MATLAB file and the translation map to Python equivalents,
see [`docs/summary.md`](docs/summary.md).

---

## License

Photonator is open-source software. See repository history for authorship.
