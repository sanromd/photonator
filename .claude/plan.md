# Photonator Development Status

## Context

Plan for Phase 2a (repo housekeeping): archive MATLAB `.m` files to a dedicated branch, migrate `.mat` data files to HDF5 in a `data/` directory, update Python callers. Informed by `docs/summary.md` and `docs/architecture.md`.

Working branch: `claude/refactor-photonator-python-nnre4`

---

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
| 6 | Advanced media (Oil dispersion, FluorescentMedium) | 📋 Planned — stubs in place |

---

## What Is Done (v2.0.0)

**Core engine**
- Vectorized NumPy propagation loop (`core/propagation.py`): exponential free-path sampling, weight-implicit absorption, Russian roulette, boundary intersection — all array ops over `PhotonBatch` (N×8 float64).
- `Simulation` orchestrator (`simulation.py`): n_batches × n_photons, pluggable beam/medium/phase_fn/receiver, CPU/GPU dispatch.
- `Receiver` (`core/receiver.py`): Fresnel transmission, TIR test, aperture, FOV cone, Welford online stats.

**Beams** — `GaussianBeam` fully implemented (Rayleigh CDF, thin-lens divergence); `LaguerreGaussianBeam`, `BesselBeam`, `AiryBeam` are class stubs with physics documented but sampling not implemented.

**Phase functions** — `HenyeyGreenstein` (analytical CDF), `PetzoldPhaseFunction` (tabulated ocean VSF from `.mat`), `MiePhaseFunction` (tabulated). `TwoTermHGPhaseFunction` and `UserDefinedPhaseFunction` are documented but absent from codebase.

**Media** — `Water` (Pope & Fry), `Oil`, `Brine` (Quan & Fry 1995), `MixtureMedium`, `LayeredMedium`/`GradientMedium` — all fully implemented.

**GPU** — Numba CUDA JIT kernel (`gpu/propagation_numba.py`) and CuPy vectorized backend (`gpu/propagation_cupy.py`); automatic fallback to NumPy CPU.

**I/O** — HDF5 (`io/hdf5.py`), `.mat` loader (`io/mat_loader.py`).

**Tests** — 38 pytest tests: Beer-Lambert exact check, HG KS-test, receiver geometry, Snell's law, MATLAB comparison scaffold.

**Docs** — 7 reference docs complete: `mcns.md`, `physics.md`, `architecture.md`, `materials.md`, `validation.md`, `gpu_setup.md`, `summary.md`.

---

## What Is Planned (Next Phases)

### Phase 2a — Repo Housekeeping

#### 2a-1: Archive MATLAB source files

**Exact .m files at repo root (21 files):**
`beamProfile.m`, `divergence.m`, `gaussianBeam.m`, `generate_scatter.m`, `loadStartFTP.m`, `lookupCDF.m`, `mc_func_Berrocal.m`, `mc_func_r6.m`, `mc_mainloop_rev4.m`, `mc_rec_Berrocal.m`, `mc_rec_r4.m`, `mc_rec_r5.m`, `normalizeAnnularHistogram.m`, `parsave.m`, `parseOutputFiles.m`, `petzold_empirical.m`, `photon_sim_Berrocal.m`, `photon_sim_r4.m`, `photon_sim (USS Skipjack's conflicted copy 2010-02-26).m`, `saveDataFTP.m`, `standaloneReceiver.m`, `startingDistribution.m`, `stat_test.m`, `test_mc_func.m`, `test_statistics.m`, `verify_vsf_lookup_table.m`, `weightedhistc.m`

**Steps (on master branch):**
1. `git checkout -b archive/matlab-original` — branch off current HEAD
2. `git rm *.m` — stage removal of all .m files (do NOT touch .mat)
3. `git commit -m "Archive MATLAB source files to archive/matlab-original"`
4. `git tag matlab-archive` — permanent tag on branch tip
5. `git checkout claude/refactor-photonator-python-nnre4` — return to working branch
6. `git rm *.m && git commit` — remove .m files from working branch too (they are preserved on archive branch)

#### 2a-2: Migrate `.mat` data files → HDF5 in `data/`

**Exact .mat files (4 files):**
- `petzold_ocean.mat` — variable `petzold_ocean`, shape (K, 4): [angle_rad, harbor_vsf, coastal_vsf, clear_vsf]
- `petzold_data_maalox_orig.mat` — Petzold-methodology Maalox VSF
- `maalox_alan_orig.mat` — A. Laux independent Maalox VSF
- `widemann_maalox.mat` — Widemann Maalox VSF, two-column [angle_rad, vsf]

**Steps:**
1. Create `scripts/convert_mat_to_hdf5.py` — uses `scipy.io.loadmat` → `h5py`; preserves all variable names as HDF5 datasets; outputs to `data/`
2. Run the script: `python scripts/convert_mat_to_hdf5.py`
3. Verify output: 4 `.h5` files in `data/`
4. Update `photonator/phase_functions/petzold.py`: change `_DATA_DIR` to point to `data/` and load `.h5` via `h5py` instead of `.mat` via `load_mat_array`
5. Update `photonator/io/mat_loader.py`: add `load_hdf5_array(path, variable)` function alongside existing `load_mat_array`
6. `MiePhaseFunction` already handles `.h5` — no change needed
7. `git rm *.mat && git commit` — remove .mat files after verification

---

### Phase 2b — micromamba Environment YAML
- Add `environment.yml` at repo root.
- Channels: `conda-forge` (primary), `nvidia` (for CUDA packages).
- Core deps: `python=3.11`, `numpy`, `scipy`, `h5py`, `pytest`, `ruff`, `mypy`.
- Optional GPU group: `numba`, `cudatoolkit`, `cupy`.
- Usage: `micromamba env create -f environment.yml && micromamba activate photonator`.
- Can be done independently of 2a.

---

### Phase 2c — Update README.md
- Depends on 2a (new `data/` structure, archive branch + tag) and 2b (env YAML).
- Describe new repo structure: `photonator/`, `data/`, `docs/`, `tests/`, `scripts/`.
- Quick-start section: micromamba env setup (`environment.yml`), then the existing `Simulation` example from `CLAUDE.md`.
- Historical MATLAB section: explain the original 2009–2011 MATLAB codebase, how to access it (`git checkout archive/matlab-original` or `git checkout matlab-archive`), and the tag `matlab-archive` for discoverability.
- Link to `docs/summary.md` for the full MATLAB-to-Python translation map.

---

### Phase 3 — Inelastic Scattering
- `emission_wavelength_nm` and `inelastic_yield` stubs exist in `AbstractPhaseFunction` and `AbstractMedium` (raise `NotImplementedError`).
- Need: Stokes-shift fluorescence emission, Raman scattering, spectral weight re-routing in propagation loop.

### Phase 4 — Spectral / Wideband
- Architecture hook documented in `physics.md §10` and `materials.md`.
- Need: per-photon wavelength array, per-wavelength µ_a/µ_s lookup, `BroadbandGaussianBeam`, spectral power accumulation in `SimulationResult`.

### Phase 5 — Complete Beam Profiles
- `LaguerreGaussianBeam`, `BesselBeam`, `AiryBeam` need `initialize()` sampling implemented (rejection sampling patterns documented in `architecture.md`).

### Phase 6 — Advanced Media
- `OilMedium` Kramers-Kronig dispersion stub.
- `FluorescentMedium` (depends on Phase 3 inelastic_yield).

---

## Verification (how to confirm current state)

```bash
cd /home/nuburu/sandbox/dev/photonator
pytest tests/ -q          # should show 38 passing
python -c "from photonator import Simulation; print('import OK')"
```

## Verification for Phase 2a

```bash
# After 2a-1: confirm .m files gone from working branch, present on archive branch
git ls-files "*.m"                        # should be empty on working branch
git show archive/matlab-original:mc_func_r6.m | head -5   # should show MATLAB code
git tag | grep matlab                     # should show matlab-archive

# After 2a-2: confirm .h5 files created and PetzoldPhaseFunction loads correctly
ls data/*.h5                              # 4 files expected
python -c "
from photonator.phase_functions import PetzoldPhaseFunction
import numpy as np
pf = PetzoldPhaseFunction('clear')
angles = pf.sample(1000, np.random.default_rng(42))
print('OK, sample mean:', angles.mean())
"
pytest tests/ -q                          # all 38 still passing
```
