# Changelog

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
