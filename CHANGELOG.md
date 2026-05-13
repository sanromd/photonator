# Changelog

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
