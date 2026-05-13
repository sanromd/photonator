# MATLAB Codebase Summary

## Overview

Photonator is a Monte Carlo photon-transport simulator originally written in MATLAB, modeling the propagation of light through scattering and absorbing aquatic media (seawater, harbor water, coastal water, Maalox suspensions used as lab surrogates). The simulator implements the standard scalar radiative-transfer Monte Carlo method: photon packets are launched from a source, propagate exponentially-distributed free-path lengths determined by the total attenuation coefficient, scatter according to a tabulated volume-scattering function (VSF), lose weight through implicit absorption at each step, and are tested for detection at a planar receiver.

The codebase grew incrementally from approximately 2009–2011, accumulating revision-numbered files (`mc_func_r6`, `mc_rec_r5`, etc.), alternative implementations (`_Berrocal`, `_rev4`), and utility scripts. The Python refactor ("photonator") replaces the per-photon MATLAB for-loop with fully vectorized NumPy/CuPy operations, restructures the code into a package with pluggable beam, medium, and receiver classes, and adds GPU support via Numba and CuPy.

---

## File Inventory

### Core Simulation Files

| File | Purpose |
|------|---------|
| `mc_func_r6.m` | Core Monte Carlo propagation loop. Initializes N×8 photon array, samples Beer-Lambert path lengths, performs binary-search CDF lookup for scatter angle, applies direction-cosine rotation, implements roulette termination, and returns received-photon statistics. |
| `mc_rec_r5.m` | Receiver module. Applies water→air Fresnel transmission, critical-angle rejection, aperture and FOV filtering with two-term Gaussian angular weighting, and accumulates Welford online mean/variance statistics for angle, distance, and weight. |
| `photon_sim_r4.m` | Top-level orchestrator script. Sets all physical parameters (c, a, b, albedo, receiver geometry, beam parameters), calls `generate_scatter` to build the CDF, loops `num_sims` times calling `mc_func_r6` then `mc_rec_r5`, and optionally saves results and sends email notification. |
| `beamProfile.m` | Gaussian beam source. Samples photon starting positions from a Gaussian radial CDF (`r = w0 * sqrt(-log(1-U))`), applies thin-lens ray-matrix transform to set divergence, assigns direction cosines. |
| `generate_scatter.m` | Phase-function CDF builder. Accepts `'measured'` (loads Petzold or Maalox VSF data files) or `'calc'` (HG formula or Haltrin empirical) mode. Integrates `VSF(theta)*sin(theta)` via `cumtrapz`, normalizes to [0, 1]. |
| `lookupCDF.m` | Binary search in CDF array. Given a uniform random variate, returns the bracketing index used for linear interpolation of the scattering angle. To be replaced by `np.searchsorted` in the Python version. |
| `petzold_empirical.m` | Script implementing the Haltrin (1997) empirical phase-function formula for harbor, coastal, and clear water types. Computes the normalized CDF and intermediate variables (q, k1–k5) from b and albedo. |
| `weightedhistc.m` | Weighted histogram with left/right edge convention, analogous to MATLAB's `histc` but accumulating per-bin sums of provided weights. Equivalent to `np.histogram(vals, bins=edges, weights=weights)`. |

### Older/Alternative Versions

| File | Purpose |
|------|---------|
| `mc_func_Berrocal.m` | Alternative MC core following Berrocal et al. methodology; different boundary conditions and scattering update. |
| `mc_rec_Berrocal.m` | Receiver matched to `mc_func_Berrocal`. |
| `mc_mainloop_rev4.m` | Revision-4 main loop; superseded by `mc_func_r6`. |
| `mc_rec_r4.m` | Revision-4 receiver; superseded by `mc_rec_r5`. |
| `photon_sim_Berrocal.m` | Orchestrator for the Berrocal variant. |
| `photon_sim (USS Skipjack's conflicted copy 2010-02-26).m` | Dropbox conflict backup of an earlier `photon_sim`; kept for historical reference only. |

### Testing and Analysis Scripts

| File | Purpose |
|------|---------|
| `divergence.m` | Standalone script computing beam divergence angle from waist and focal parameters. |
| `gaussianBeam.m` | Reference Gaussian beam demo; plots irradiance profile and propagation. |
| `startingDistribution.m` | Visualizes the spatial and angular starting distribution produced by `beamProfile`. |
| `standaloneReceiver.m` | Runs `mc_rec_r5` on previously saved photon data files without rerunning the MC loop. |
| `stat_test.m` | Statistical sanity checks on simulation output (mean, variance, histograms). |
| `test_mc_func.m` | Unit-style test for `mc_func_r6` with known analytic cases. |
| `test_statistics.m` | Compares statistics across multiple simulation runs to verify convergence. |
| `verify_vsf_lookup_table.m` | Verifies that the CDF lookup and interpolation reproduce the expected scattering-angle distribution. |
| `normalizeAnnularHistogram.m` | Corrects radial histograms for annular area bias (divides by the annular ring area). |

### I/O Utilities

| File | Purpose |
|------|---------|
| `parsave.m` | Parallel-safe save wrapper: saves per-batch result files named by simulation index, avoiding Parallel Toolbox file-access race conditions. |
| `parseOutputFiles.m` | Post-processing script that reads and concatenates per-batch `.mat` output files produced by `parsave`. |
| `saveDataFTP.m` | Uploads a result archive to an FTP server (used for AWS-to-local data retrieval). |
| `loadStartFTP.m` | Downloads input data files from FTP before a simulation run on a remote machine. |

### Data Files

| File | Contents |
|------|---------|
| `petzold_ocean.mat` | Petzold (1972) VSF measurements. Four columns: `angle_rad`, harbor VSF (1/m/sr), coastal VSF, clear-ocean VSF. Angular resolution approximately 0.1 degrees at forward angles. |
| `petzold_data_maalox_orig.mat` | VSF of Maalox antacid suspension measured following Petzold methodology. Used as a controlled lab scatterer. |
| `maalox_alan_orig.mat` | Independent Maalox VSF dataset measured by A. Laux; provides a second Maalox reference for cross-validation. |
| `widemann_maalox.mat` | Maalox VSF from Widemann et al.; two-column format [angle_rad, VSF] covering a different angular range. |

---

## Detailed Descriptions of Core Files

### `mc_func_r6.m` — Core MC Propagation Loop

**Signature:**
```matlab
function [total_time, total_rec_power, total_rec_packets, rec_loc_final, ...
          total_rec_dist, rec_weights] = ...
    mc_func_r6(num_photons, scattering_events, c, a, receiver_z, ...
               cdf_scatter, angle, init_angle, init_angle2, ...
               beamDiverg, beamWidth, wallAbsorption)
```

**Photon array layout:** `photon(i, 1:8)` stores [x, y, z, ux, uy, uz, weight, status], where status = 1 (active), 0 (received), -1 (terminated).

**Initialization:** Calls `beamProfile` to populate starting (x, y) positions and (ux, uy, uz) direction cosines. Weight = 1; status = 1 for all photons.

**Main loop:** Iterates `while photonsRemaining > 0`. On each pass, a `num_photons x 3` random matrix is generated: column 1 for free-path sampling, column 2 for CDF lookup, column 3 for azimuthal angle phi.

**Path length sampling:** `r = -inv_c * log(rand_array(i,1))`, i.e., exponential with mean 1/c (Beer-Lambert).

**Scatter-angle lookup:** Manual binary search on `cdf_scatter`; the converged bracket index `k` is used to linearly interpolate `theta` from the `angle` vector.

**Receiver plane test:** If `z + z_step >= receiver_z`, the photon is ray-traced to the plane. A spatial window (+-3 m in x and y) may reject edge photons. Reception position, direction cosines, weight, and path length are recorded.

**Propagation (scatter branch):** Position is advanced by `(r*ux, r*uy, r*uz)`. Weight is multiplied by the survival probability `b/c`. Roulette is applied when `weight < min_power`: the photon is terminated with probability `(rouletteConst-1)/rouletteConst` or has its weight multiplied by `rouletteConst`.

**Direction update:** Standard direction-cosine rotation. When `|uz| > 1 - 1e-12`, uses the degenerate-case formula to avoid division by zero. Otherwise:

```
ux' = (sin(theta) / sqrt(1-uz^2)) * (ux*uz*cos(phi) - uy*sin(phi)) + ux*cos(theta)
uy' = (sin(theta) / sqrt(1-uz^2)) * (uy*uz*cos(phi) + ux*sin(phi)) + uy*cos(theta)
uz' = -sin(theta)*cos(phi) * sqrt(1-uz^2) + uz*cos(theta)
```

The unit vector is renormalized if it deviates from unity by more than 1e-11.

**Outputs:** Elapsed time, total received power (raw sum of weights at receiver plane), packet count, received-photon location/direction array, per-photon path lengths, per-photon weights.

---

### `mc_rec_r5.m` — Receiver Module

**Signature:**
```matlab
function [power, ph_cnt, angle_mean, angle_var, dist_mean, dist_var, ...
          weight_mean, weight_var, reflected, distances, angles, weights] = ...
    mc_rec_r5(a, rec_loc_final, total_rec_dist, rec_weights, ...
              rec_pos, rec_aperture, rec_fov, numTxPhotons)
```

**Critical-angle test:** `critAngCos = sqrt(1 - (n_air/n_water)^2)`. Photons with `uz < critAngCos` undergo total internal reflection and are rejected (the `reflected` counter is incremented).

**Fresnel transmission:** For photons that pass the critical-angle test:

```
cosExitAng = sqrt(1 - (n_water/n_air)^2 * (1 - uz^2))
rp = (uz - n_water*cosExitAng) / (uz + n_water*cosExitAng)
rs = (cosExitAng - n_water*uz) / (cosExitAng + n_water*uz)
R  = (rp^2 + rs^2) / 2
T  = 1 - R
```

Weight is multiplied by T; `uz` is updated to `cosExitAng`.

**Aperture and FOV test:** `distance = sqrt((ph_x-rx_x)^2 + (ph_y-rx_y)^2) <= radius` AND `uz >= cos(fov/2)`.

**Angular weighting:** Accepted photons are weighted by the two-term Gaussian:

```
fovWeight = A1*exp(-((theta-B1)/C1)^2) + A2*exp(-((theta-B2)/C2)^2)
```

with hard-coded constants A1=0.7985, B1=0.0187, C1=0.03437, A2=0.7121, B2=-0.02337, C2=0.03117.

**Welford online statistics:** Mean and variance for arrival angle, path length, and weighted power are accumulated in a single pass without storing all individual values.

**Output normalization:** `weight_mean` and `weight_var` are normalized to the total transmitted photon count `numTxPhotons` at the end of the loop, giving estimates of per-photon statistics.

---

### `photon_sim_r4.m` — Orchestrator Script

Sets all physical parameters at the top of the script (not as function arguments). Notable choices at default settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_photons` | 1e5 | Photons per batch |
| `num_sims` | 10 | Number of batches |
| `c` | 0.15 m⁻¹ | Clear ocean attenuation |
| `albedo` | 0.25 | Fraction scattered (b/c) |
| `receiver_z` | attenuationLength / c | Fixed attenuation lengths away |
| `attenuationLength` | 16 | Optical depths |
| `beamWidth` | 0.001 m | 1 mm half-width |
| `beamDiverg` | 0.00075 rad | Half-angle at waist |
| `rec_aperture` | 0.8 m | Receiver diameter |
| `rec_fov` | pi/2 rad | 90 degree full FOV |

The AWS/Linux branch reads user-data from the EC2 metadata service to detect auto-run mode, configures Gmail SMTP, and emails results on completion. The script includes vestigial parallel-pool code that is commented out.

---

### `beamProfile.m` — Gaussian Beam Source

Generates initial photon positions and direction cosines for a paraxial Gaussian beam.

**Radial CDF:** The irradiance profile of a Gaussian beam is `I(r) proportional to exp(-2r^2/w0^2)`. The CDF of the corresponding radial distribution is `F(r) = 1 - exp(-r^2/w0^2)`, inverted by:

```matlab
radius = beamWaist * sqrt(-log(1 - randVals))
```

**Thin-lens divergence:** A thin lens with inverse focal length `invF = -diverg/beamWaist` maps the position at the beam waist to a polar divergence angle:

```
divAng = -invF * radius = (diverg/beamWaist) * radius
```

This ensures that the marginal ray at `r = w0` has the design half-angle divergence.

**Direction cosines:**

```
uz = cos(divAng),  sin_uz = sqrt(1 - uz^2)
ux = sin_uz * cos(phi),  uy = sin_uz * sin(phi)
```

where phi is uniform on [0, 2*pi]. Starting (x, y) coordinates are `radius * (cos(phi), sin(phi))`.

---

### `generate_scatter.m` — Phase-Function CDF Builder

Builds the scattering-angle CDF used by `mc_func_r6` for inverse-transform sampling.

**Measured mode (`func_type = 'measured'`):** Loads one of the VSF data files (Petzold harbor, coastal, clear; various Maalox datasets; Mie 1 micron). Applies:

```matlab
cdf_scatter = cumtrapz(angle_rad, VSF .* sin(angle_rad))
cdf_scatter = cdf_scatter ./ max(cdf_scatter)
```

The `sin(theta)` factor converts the VSF (a function of solid angle `dOmega = sin(theta) dtheta dphi`) into a 1D PDF over theta after integrating out phi.

**Calculated HG mode (`func_type = 'calc', water_cond = 'hg'`):** Evaluates the Henyey-Greenstein formula on a non-uniform grid (0.01 degree steps to 10 degrees, then 0.1 degree steps to 180 degrees) and applies the same cumtrapz normalization.

**Calculated Haltrin mode:** Uses the empirical polynomial-exponential formula from Haltrin (1997) with coefficients derived from the scattering coefficient b and albedo omega_0.

---

### `lookupCDF.m` — Binary Search CDF Lookup

Implements a bisection search over the `CDF` array to find the index straddling a given uniform random value. The loop terminates when `maxIndex < minIndex` or an exact match is found. The result is the `midIndex` at convergence, which `mc_func_r6` then uses for linear interpolation.

**Known issue:** The exact-match branch (`value == CDF(midIndex)`) is almost never triggered in floating-point arithmetic, so the loop always runs to convergence by bracket narrowing, which is correct but slightly wasteful. `np.searchsorted` replaces this with an O(log N) C-level call and handles edge cases cleanly.

---

### `petzold_empirical.m` — Haltrin Empirical Phase Function

Script (not a function) implementing Haltrin (1997) for the volume-scattering function. The phase function uses a degree-5 polynomial in sqrt(theta) with albedo-dependent coefficients k1 through k5 and a leading coefficient q that depends on sqrt(b):

```
q  = 2.598 + 17.748*sqrt(b) - 16.722*b + 5.932*b*sqrt(b)
k1 = 1.188 - 0.688*omega_0
k2 = 0.1*(3.07 - 1.90*omega_0)
k3 = 0.01*(4.58 - 3.02*omega_0)
k4 = 0.001*(3.24 - 2.25*omega_0)
k5 = 0.0001*(0.84 - 0.61*omega_0)
```

Theta is in degrees. The script also computes the normalization integral `0.5 * trapz(theta, p(theta)*sin(theta))` to verify phase-function normalization.

---

### `weightedhistc.m` — Weighted Histogram

Provides a bin-accumulation loop over `edges` that sums `weights(i)` into the bin containing `vals(i)`, with left-inclusive or right-inclusive edge conventions. Both conventions handle the terminal edge as a single-point match. The equivalent NumPy call is `np.histogram(vals, bins=edges, weights=weights)`.

---

## Known Limitations of the MATLAB Code

1. **Per-photon serial loop:** The inner `for i = 1:num_photons` loop in `mc_func_r6` iterates serially over photons. MATLAB's JIT can partially accelerate this, but it is fundamentally not vectorized. The Python port replaces this with NumPy array operations on the full active-photon batch simultaneously.

2. **Manual binary search:** `lookupCDF.m` implements a bisection loop in interpreted MATLAB. On each scatter event, O(log N) MATLAB iterations are executed for each photon. `np.searchsorted` performs the same operation in compiled C with SIMD and cache-friendly memory access.

3. **No GPU support:** The MATLAB code has no parallel toolbox GPU calls. The Python port adds Numba CUDA and CuPy backends.

4. **Single beam type:** Only the paraxial Gaussian beam in `beamProfile.m` is wired into `mc_func_r6`. Other profiles (flat-top, Bessel, user-defined) require code changes. The Python port uses an `AbstractBeam` interface.

5. **No layered media:** The code assumes a homogeneous medium with uniform c, a, b throughout the propagation volume. Stratified ocean models with depth-varying optical properties require a redesigned step-length integration, which is absent.

6. **Hardcoded water IOR:** `nWater = 1.33` appears as a literal in multiple files. A wavelength-dependent Sellmeier model (e.g., Quan and Fry 1995) is not implemented.

7. **Single wavelength per run:** Optical properties are scalars. Broadband or multi-spectral sources require re-running the entire simulation for each wavelength, then summing.

8. **Hardcoded Fresnel/FOV constants:** The two-term Gaussian FOV weighting constants in `mc_rec_r5` are numerical literals with no documented origin or uncertainty. They appear to be fits to a specific detector's angular response.

9. **No inelastic scattering:** Fluorescence and Raman scattering, important for some ocean sensing applications, are not modeled. Stub extension points are reserved in the Python architecture.

10. **Global state in orchestrator:** `photon_sim_r4.m` is a script rather than a function, relying on MATLAB workspace variables. This makes automated parameter sweeps and unit testing difficult.

---

## MATLAB to Python Translation Map

| MATLAB construct | Python equivalent | Notes |
|-----------------|-------------------|-------|
| `photon = zeros(N, 8)` | `photon = np.zeros((N, 8), dtype=np.float64)` | N x 8 photon state array |
| `rand(N, 3)` | `rng.random((N, 3))` | NumPy default RNG |
| `r = -inv_c * log(rand(...))` | `r = rng.exponential(inv_c, N)` | Exponential samples |
| `cumtrapz(angle, VSF .* sin(angle))` | `np.cumulative_trapezoid(vsf * np.sin(angle), angle)` | SciPy or NumPy |
| `cdf ./ max(cdf)` | `cdf /= cdf[-1]` | Normalize CDF |
| `lookupCDF(cdf, val)` (binary search) | `np.searchsorted(cdf, val)` | Vectorized, returns array |
| Linear interpolation of theta | `np.interp(u, cdf, angle)` | Vectorized inverse CDF |
| `sqrt(1 - uz^2)` | `np.sqrt(1 - uz**2)` | sin(theta) from cos(theta) |
| Direction-cosine update (near-singular) | Masked array update with `np.where` | Handle `|uz| ~= 1` branch |
| Direction-cosine update (general) | Vectorized formula on active-photon slice | One array op per component |
| `weightedhistc(vals, weights, edges)` | `np.histogram(vals, bins=edges, weights=weights)` | Built-in |
| Welford online stats loop | Manual NumPy accumulation or `welford` package | One pass, no per-sample storage |
| `parsave(...)` | `np.save(path, data)` or HDF5 via h5py | Per-batch save |
| `parseOutputFiles(...)` | `np.load` or `h5py.File` in a loop | Concatenate batches |
| `matlabpool` | `multiprocessing.Pool` or `concurrent.futures` | CPU parallelism |
| GPU (absent) | `import cupy as cp` or `@cuda.jit` | CuPy and Numba backends |
| `nWater = 1.33` (literal) | `scipy.constants` plus Quan and Fry 1995 | Wavelength-dependent IOR |
| Script with global workspace | `Simulation` class with `run()` method | Testable, parameterized |
| `beamProfile.m` (one type) | `GaussianBeam(AbstractBeam)` | Pluggable via interface |
| `generate_scatter.m` (switch) | `PetzoldPhaseFunction(AbstractPhaseFunction)` | Pluggable via interface |
