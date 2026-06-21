# Photonator Development Plan

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 1 | MATLAB implementation (2009–2011) | ✅ Complete |
| 2 | Python rewrite + GPU backends (v2.0.0, 2026-05-13) | ✅ Complete |
| 2a | Repo housekeeping (archive MATLAB, migrate data files) | ✅ Complete |
| 2b | micromamba environment YAML | ✅ Complete |
| 2c | Update README.md | ✅ Complete |
| 3 | Inelastic scattering (fluorescence, Raman) | 📋 Planned — stubs in place |
| 4 | Spectral / wideband support | 📋 Planned — stubs in place |
| 5 | Full beam profile implementations (LG, Bessel, Airy) | 📋 Planned — stubs in place |
| 6 | Advanced media (Oil dispersion, FluorescentMedium) | 📋 Planned — stubs in place |

---

## What Is Done (v2.0.0)

**Core engine**
- Vectorized NumPy propagation loop (`core/propagation.py`): exponential free-path sampling, weight-implicit absorption, Russian roulette, boundary intersection — all array ops over `PhotonBatch` (N×8 float64).
- `Simulation` orchestrator (`simulation.py`): n_batches × n_photons, pluggable beam/medium/phase_fn/receiver, CPU/GPU dispatch.
- `Receiver` (`core/receiver.py`): Fresnel transmission, TIR test, aperture, FOV cone, Welford online stats.

**Beams** — `GaussianBeam` fully implemented (Rayleigh CDF, thin-lens divergence); `LaguerreGaussianBeam`, `BesselBeam`, `AiryBeam` are class stubs with physics documented but sampling not yet implemented.

**Phase functions** — `HenyeyGreenstein` (analytical CDF), `PetzoldPhaseFunction` (tabulated ocean VSF from HDF5), `MiePhaseFunction` (tabulated).

**Media** — `Water` (Pope & Fry), `Oil`, `Brine` (Quan & Fry 1995), `MixtureMedium`, `LayeredMedium`/`GradientMedium` — all fully implemented.

**GPU** — Numba CUDA JIT kernel (`gpu/propagation_numba.py`) and CuPy vectorized backend (`gpu/propagation_cupy.py`); automatic fallback to NumPy CPU.

**I/O** — HDF5 (`io/hdf5.py`), mat/HDF5 loader (`io/mat_loader.py`).

**Tests** — 38 pytest tests: Beer-Lambert exact check, HG KS-test, receiver geometry, Snell's law, MATLAB comparison scaffold.

**Docs** — 7 reference docs: `mcns.md`, `physics.md`, `architecture.md`, `materials.md`, `validation.md`, `gpu_setup.md`, `summary.md`.

**Repo housekeeping (2a–2c)**
- MATLAB `.m` source files archived on `archive/matlab-original` branch (tag `matlab-archive`).
- `.mat` VSF data files converted to `data/*.h5` (HDF5); `.mat` files removed from working branch.
- `environment.yml` added for micromamba setup.
- `README.md` updated to reflect current Python-based structure.

---

## Planned Phases

### Phase 3 — Inelastic Scattering

`emission_wavelength_nm` and `inelastic_yield` stubs exist in `AbstractPhaseFunction` and `AbstractMedium`.

Needed:
- Stokes-shift fluorescence emission model
- Raman scattering cross-section and angle distribution
- Spectral weight re-routing in the propagation loop (`core/propagation.py`)

### Phase 4 — Spectral / Wideband

Architecture hook documented in `physics.md §10` and `materials.md`.

Needed:
- Per-photon wavelength array in `PhotonBatch`
- Per-wavelength µ_a / µ_s lookup in `AbstractMedium`
- `BroadbandGaussianBeam` (spectral power distribution)
- Spectral power accumulation in `SimulationResult`

### Phase 5 — Complete Beam Profiles

`LaguerreGaussianBeam`, `BesselBeam`, and `AiryBeam` exist as class stubs. The physics and rejection-sampling patterns are documented in `architecture.md`. `initialize()` sampling needs to be implemented for each.

### Phase 6 — Advanced Media

- `OilMedium`: Kramers-Kronig dispersion for wavelength-dependent IOR (stub).
- `FluorescentMedium`: bulk fluorophore model (depends on Phase 3 `inelastic_yield`).
