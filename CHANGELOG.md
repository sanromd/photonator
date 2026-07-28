# Changelog

## [2.1.0] - 2026-07-28

### Added — Phase 3: inelastic scattering (fluorescence)
- Fluorescence conversion in `propagate_cpu`: media exposing the inelastic
  hooks (`inelastic_yield`, `emission_wavelength_nm`,
  `mu_a_fluorophore_per_m`) convert photons at interactions with
  probability μ_a,f·QY / (μ_s + μ_a,f·QY); the implicit-capture survival
  factor grows accordingly. New `FLUORESCED` photon status.
- Two-pass emission propagation in `Simulation.run`: converted photons are
  re-emitted isotropically at the emission wavelength and propagated in
  the host medium (`FluorescentMedium.emission_medium()`), single
  generation, with path length carried over. Detected on a separate
  receiver channel: `SimulationResult.fluorescent_power`,
  `fluorescent_packets`, `fluoresced_photons`, `emission_wavelength_nm`.

### Added — Phase 4: spectral / wideband support
- `AbstractMedium.at_wavelength(wavelength_nm)` API (identity by default);
  overridden by `TabulatedMedium` and `Water`.
- Pope & Fry (1997) pure-water absorption table (400–700 nm) in
  `media/water.py` with `pope_fry_absorption_m_inv()`; `Water.at_wavelength`
  interpolates it.
- `SpectralSimulation` / `SpectralResult` (`photonator/spectral.py`):
  wavelength-binned wideband sources — one monochromatic run per bin with
  the medium evaluated at the bin wavelength, weighted by the source
  spectrum.

### Added — layered propagation (previously suggested enhancement)
- `propagate_layered_cpu` (`core/propagation_layered.py`): per-layer
  attenuation/albedo/roulette thresholds, memoryless step re-draw at
  boundaries, Snell refraction + unpolarised Fresnel reflection + TIR at
  interfaces, per-layer phase functions via new `Layer.phase_fn` field.
  `Simulation` dispatches automatically for `LayeredMedium` (CPU backend).
- `LayeredMedium.layers` public property; `Receiver.clone()`.

### Fixed
- Pope & Fry 532 nm water absorption default corrected from 0.0088 to
  0.0442 m⁻¹ (the old value belonged near 450 nm); `Brine` baseline synced.

### Changed
- `Simulation` raises `NotImplementedError` for `GradientMedium` (needs
  adaptive sub-stepping; discretise into a `LayeredMedium` instead) and for
  layered/fluorescent runs on GPU backends.
- Propagation accumulates path length in `PhotonBatch._path_length_m` so
  multi-pass runs carry distance across passes.

## [2.0.1] - 2026-07-28

### Fixed
- **Propagation index-remapping bug**: after mid-loop terminations (z<0,
  zero-weight, or roulette filters), surviving photons were rotated from
  *other photons'* previous directions, breaking per-photon angular
  continuity. Filter positions are now carried through every filter in
  lockstep (`sel` array in `core/propagation.py`). Regression test:
  `tests/test_propagation.py::test_direction_continuity_under_terminations`.
- Missing `import math` in `gpu/propagation_numba.py` (kernel raised
  `NameError` on any CUDA machine).
- `pip install -e .` broken by setuptools flat-layout auto-discovery;
  packages now declared explicitly in `pyproject.toml`.
- Deprecated `datetime.utcnow()` in `io/hdf5.py`.

### Changed
- `Receiver` statistics now use vectorized Welford accumulation
  (Chan et al. parallel merge) instead of a per-photon Python loop.
- New public `AbstractPhaseFunction.cdf_table()` API; GPU backends no
  longer reach into private `_cdf`/`_angles_rad` attributes.
- `validation/benchmarks.py` HG KS-test rewritten against the closed-form
  CDF of cos θ (`hg_cos_cdf`).
- Codebase is ruff-clean; physics-notation rules (N803/N806/E741)
  ignored by config. GitHub Actions CI added (ruff + pytest on 3.10/3.12).

## [2.0.0] - 2026-05-13

### Added
- Python rewrite of the original MATLAB Photonator MC simulator
- Vectorized NumPy propagation loop replacing per-photon MATLAB for-loop
- `PhotonBatch` N×8 float64 array with active-mask slicing
- `Simulation` class with pluggable beam / medium / phase-function / receiver
- `GaussianBeam` with Rayleigh-CDF radius sampling and thin-lens divergence
- `LaguerreGaussianBeam`, `BesselBeam`, `AiryBeam` profiles
- `HenyeyGreensteinPhaseFunction` with analytical CDF inversion
- `PetzoldPhaseFunction` loading measured ocean VSF from `.mat` data files
- `MiePhaseFunction` loading tabulated Mie VSF
- `Water`, `Oil`, `Brine`, `MixtureMedium` material models
- `LayeredMedium` with hard-boundary, gradient, and random-spatial layer types
- `Receiver` with Fresnel water→air, critical-angle, aperture, FOV, and Welford statistics
- GPU backends: Numba CUDA JIT kernel and CuPy vectorized propagation
- HDF5 output via `photonator.io.hdf5`
- `.mat` data-file loader via `photonator.io.mat_loader`
- Validation benchmarks: Beer-Lambert, HG KS-test, MATLAB comparison
- Full test suite under `tests/`
- `CLAUDE.md` agent guide, `docs/` reference documentation

### Preserved
- All original MATLAB source files unchanged

## [1.x] - 2010–2011

Original MATLAB implementation by William Cox (PhD research, underwater optical comms).
