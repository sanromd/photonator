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
| 5 | Beam profile tests + comparison notebook | ✅ Complete |
| 6 | Advanced media (DispersiveOil, FluorescentMedium) | ✅ Complete |

---

## What Is Done (v2.0.0)

**Core engine**
- Vectorized NumPy propagation loop (`core/propagation.py`): exponential free-path sampling, weight-implicit absorption, Russian roulette, boundary intersection — all array ops over `PhotonBatch` (N×8 float64).
- `Simulation` orchestrator (`simulation.py`): n_batches × n_photons, pluggable beam/medium/phase_fn/receiver, CPU/GPU dispatch.
- `Receiver` (`core/receiver.py`): Fresnel transmission, TIR test, aperture, FOV cone, Welford online stats.

**Beams** — all four fully implemented: `GaussianBeam` (Rayleigh CDF, thin-lens divergence), `LaguerreGaussianBeam` (rejection sampling, LG profile), `BesselBeam` (J₀² rejection sampling), `AiryBeam` (2D separable Airy rejection sampling). 23 unit tests in `tests/test_beams.py`; `notebooks/03_beam_profiles.ipynb` shows side-by-side intensity maps with theoretical overlays.

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

### Phase 5 — Beam Profile Tests and Comparison Notebook ✅

All three beams were already fully implemented (not stubs). Phase 5 delivered:
- `tests/test_beams.py`: 23 tests across Gaussian, LG, Bessel, and Airy covering unit direction cosines, position support bounds, profile-shape invariants (donut zero, Bessel central peak, Airy first-lobe offset), and transverse scaling.
- `notebooks/03_beam_profiles.ipynb`: side-by-side 2D intensity maps, radial profiles vs analytical curves, uz cone structure, and summary statistics table.

### Phase 6 — Advanced Media ✅

`DispersiveOil` (in `photonator/media/oil.py`):
- Cauchy/Sellmeier dispersion: n(λ) = A + B/λ_μm² + C/λ_μm⁴, calibrated at 589 nm.
- `n_at(wavelength_nm)` for stateless evaluation without mutating instance state.
- Numerical Kramers-Kronig: optional `k_wavelengths_nm` + `k_spectrum` input; integrates n(ω₀) = 1 + (2/π) P∫ ω k(ω)/(ω²−ω₀²) dω.
- Falls back to fixed n from `Oil` when `wavelength_nm` is None.

`FluorescentMedium` (`photonator/media/fluorescent.py`):
- Wraps any `AbstractMedium` host, adds `mu_a_ex_per_m` fluorophore absorption.
- Sets `inelastic_yield` = `quantum_yield` and `emission_wavelength_nm` for Phase 3 propagation loop hook.
- `from_concentration(concentration_mol_per_L, molar_extinction_L_per_mol_cm, ...)` classmethod via Beer-Lambert conversion.
- 32 tests in `tests/test_advanced_media.py`; 93 total tests passing.
