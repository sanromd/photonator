# CLAUDE.md — Photonator Agent Guide

## Scientific Goal

Photonator is a Monte Carlo photon-transport simulator for scattering/absorbing liquids
(seawater, brines, oils, mixtures). It studies how beam type, wavelength, geometry, and
material composition affect optical propagation — for applications in underwater
photonics, optical communications, and random-media research.

## Accepted Approximations

| Approximation | Scope |
|---|---|
| Paraxial beams | Divergence << 1 rad; valid for Gaussian, LG, Bessel, Airy |
| Geometric optics at interfaces | Snell + Fresnel; no boundary diffraction |
| Independent scattering | Dilute-particle limit; single-event Mie/HG per scatter |
| Elastic scattering only (Phase 1–4) | No fluorescence or Raman |
| Monochromatic per run | One wavelength per simulation; architecture spectral-ready |
| Inelastic hooks reserved | `emission_wavelength_nm`, `inelastic_yield` stubs in `AbstractPhaseFunction` and `AbstractMedium` for future wideband extension |

## Quick Start

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
from photonator.io import save_hdf5
save_hdf5(result, "output.h5")
```

## Module Map

| File | Class | Purpose |
|---|---|---|
| `photonator/simulation.py` | `Simulation`, `SimulationResult` | Top-level batch orchestrator |
| `photonator/core/photon.py` | `PhotonBatch` | N×8 float64 photon state array |
| `photonator/core/propagation.py` | `propagate_cpu` | Vectorized NumPy MC loop |
| `photonator/core/scattering.py` | `update_direction` | Direction-cosine rotation |
| `photonator/core/receiver.py` | `Receiver` | Fresnel + FOV + aperture + Welford stats |
| `photonator/beam/gaussian.py` | `GaussianBeam` | Rayleigh-CDF + thin-lens divergence |
| `photonator/beam/laguerre.py` | `LaguerreGaussianBeam` | LG_p^l rejection sampling |
| `photonator/beam/bessel.py` | `BesselBeam` | J0(k_r·r) rejection sampling |
| `photonator/beam/airy.py` | `AiryBeam` | Ai(x/x0)·Ai(y/y0) rejection sampling |
| `photonator/phase_functions/henyey_greenstein.py` | `HenyeyGreensteinPhaseFunction` | Analytical CDF inversion |
| `photonator/phase_functions/petzold.py` | `PetzoldPhaseFunction` | Measured ocean VSF from .mat |
| `photonator/phase_functions/mie.py` | `MiePhaseFunction` | Tabulated Mie VSF |
| `photonator/media/water.py` | `Water` | Pope & Fry absorption, Petzold scatter |
| `photonator/media/oil.py` | `Oil` | Literature IOR for crude/refined/mineral |
| `photonator/media/brine.py` | `Brine` | Quan & Fry (1995) IOR for NaCl/KCl/CaCl2 |
| `photonator/media/mixture.py` | `MixtureMedium` | Volume-fraction mixing |
| `photonator/media/layers.py` | `LayeredMedium`, `GradientMedium` | Axially stratified media |
| `photonator/gpu/propagation_numba.py` | `propagate_numba` | Numba CUDA JIT kernel |
| `photonator/gpu/propagation_cupy.py` | `propagate_cupy` | CuPy vectorized backend |
| `photonator/io/hdf5.py` | `save_hdf5`, `load_hdf5` | HDF5 result I/O |
| `photonator/io/mat_loader.py` | `load_mat_array` | MATLAB .mat data loader |
| `photonator/validation/benchmarks.py` | — | Beer-Lambert, HG KS-test, power budget |
| `photonator/validation/matlab_compare.py` | — | Compare vs MATLAB output |
| `photonator/constants.py` | — | Physics constants (wraps scipy.constants) |

## Recipes

**Add a new beam**: subclass `AbstractBeam` in `photonator/beam/`, implement
`initialize(n_photons, rng) → PhotonBatch`.  Beam sets `x_m`, `y_m`, `ux`, `uy`, `uz`; leave
`z_m=0`, `weight=1`, `status=ACTIVE`.

**Add a new material**: subclass `AbstractMedium` in `photonator/media/`, implement
`mu_a_per_m`, `mu_s_per_m`, `g`, `n` as `@property`.  Optionally set
`wavelength_nm` for spectral-ready models.  Register in `photonator/media/__init__.py`.

**Add a new phase function**: subclass `AbstractPhaseFunction` in
`photonator/phase_functions/`, implement `sample(n, rng) → NDArray`.  Use
`self.build_cdf(angles_rad, vsf)` + `self.sample_from_cdf(...)` for tabulated data.
Set `emission_wavelength_nm` / `inelastic_yield` stubs if adding inelastic processes.

**Run tests**: `pytest tests/ -q`

**Run MATLAB comparison**: see `docs/validation.md`

## Validation Strategy

Each simulation phase is validated against analytical benchmarks (Beer-Lambert,
HG KS-test, Snell's law) and against saved MATLAB output.  See `docs/validation.md`
for pass criteria and `photonator/validation/` for runnable scripts.

## Performance Constraints

| Target | Constraint |
|---|---|
| GPU (CUDA) | 1M photons < 60 s (RTX 3080 or equivalent) |
| CPU (NumPy) | 1M photons < 10 min |
| Memory | < 8 GB GPU VRAM for 1M photons |

## Reference Documentation

- [Monte Carlo numerics](docs/mcns.md) — path sampling, CDF, roulette, Welford stats
- [Physics equations](docs/physics.md) — Beer-Lambert, HG, Fresnel, Snell, beam profiles
- [Architecture](docs/architecture.md) — class hierarchy, data flow, extension points
- [Materials](docs/materials.md) — optical property sources, override format
- [Validation](docs/validation.md) — pass criteria, test commands
- [GPU setup](docs/gpu_setup.md) — CUDA/Numba/CuPy install guide
- [MATLAB summary](docs/summary.md) — original 30-file MATLAB codebase map
