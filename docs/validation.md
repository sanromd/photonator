# Validation Guide

This document describes the validation test suite for the photonator Python package. Each test has a documented expected result, tolerance, and pass/fail criterion. All tests are implemented under `tests/` and can be run with pytest. The MATLAB comparison test requires MATLAB `.mat` output files and is run separately.

---

## Running the Test Suite

**Unit and integration tests:**

```bash
pytest tests/ -v
```

**MATLAB comparison validation only:**

```bash
python -m photonator.validation.matlab_compare --mat-dir /path/to/matlab/output/
```

**Full validation including MATLAB comparison:**

```bash
pytest tests/ -v --matlab-dir /path/to/matlab/output/
```

Set the environment variable `PHOTONATOR_TEST_DATA` to the directory containing reference `.mat` files if the path is not passed explicitly:

```bash
export PHOTONATOR_TEST_DATA=/path/to/matlab/output/
pytest tests/ -v
```

---

## Test 1: Beer-Lambert Transmission (Pure Absorber)

**Physical basis:** In a medium with absorption coefficient a (1/m) and no scattering (b = 0), the transmitted intensity after path length L is:

```
T = exp(-a * L)
```

This is the exact analytical solution to the RTE in the limit b = 0.

**Setup:**

```python
# tests/test_beer_lambert.py
import numpy as np
from photonator import Simulation, Receiver
from photonator.beams import GaussianBeam
from photonator.media import HomogeneousMedium
from photonator.phase_functions import HenyeyGreensteinPF

def test_beer_lambert_transmission():
    a_per_m = 0.5          # 1/m absorption
    b_per_m = 0.0          # no scattering
    L_m = 2.0              # path length (m)
    expected_T = np.exp(-a_per_m * L_m)   # = exp(-1.0) ≈ 0.3679

    beam = GaussianBeam(waist_m=0.001, divergence_rad=0.0)
    medium = HomogeneousMedium(
        a_per_m=a_per_m, b_per_m=b_per_m,
        phase_function=HenyeyGreensteinPF(g=0.0)
    )
    rx = Receiver(
        position_xy=(0.0, 0.0),
        aperture_m=100.0,    # large aperture catches all photons
        fov_rad=np.pi,       # full hemisphere FOV
        fov_weighting='uniform'
    )
    sim = Simulation(
        beam=beam, medium=medium, receiver=rx,
        receiver_z_m=L_m, n_photons=200_000, n_batches=5, seed=42
    )
    result = sim.run()

    measured_T = result.mean_weight   # normalized received power per launched photon
    assert abs(measured_T - expected_T) / expected_T < 0.01, (
        f"Beer-Lambert transmission error: expected {expected_T:.6f}, "
        f"got {measured_T:.6f} ({abs(measured_T-expected_T)/expected_T*100:.2f}% error)"
    )
```

**Expected value:** T = exp(-a*L) = exp(-0.5 * 2.0) = exp(-1.0) = 0.36788.

**Tolerance:** Relative error < 1% (absolute error < 0.00368).

**Convergence note:** With N = 1,000,000 photons and b = 0, all photons reach the receiver and the statistical uncertainty is essentially zero; the test is limited by floating-point arithmetic, not Monte Carlo variance. The test should pass easily with N >= 10,000.

---

## Test 2: HG Phase Function Distribution (KS Test)

**Physical basis:** The Henyey-Greenstein phase function has an exact closed-form CDF. Angles sampled by the photonator CDF sampler should follow this distribution. A Kolmogorov-Smirnov (KS) test with null hypothesis H_0: "samples follow the HG distribution" should not be rejected at the 5% significance level.

**Setup:**

```python
# tests/test_phase_function.py
import numpy as np
from scipy import stats
from photonator.phase_functions import HenyeyGreensteinPF

def analytical_hg_cdf(theta, g):
    """Analytical CDF of the HG marginal distribution over theta."""
    cos_theta = np.cos(theta)
    # CDF(theta) = integral_0^theta p(t)*sin(t)dt / integral_0^pi p(t)*sin(t)dt
    # For HG, this simplifies to:
    numerator = (1 - g**2) / (1 + g**2 - 2*g*cos_theta)**0.5 - (1 - g)
    denominator = (1 - g**2) / (1 + g**2 + 2*g)**0.5 - (1 - g)
    return numerator / denominator

def test_hg_phase_function_ks():
    g = 0.924
    rng = np.random.default_rng(seed=0)
    pf = HenyeyGreensteinPF(g=g)

    n_samples = 100_000
    theta_samples = pf.sample_angle(n_samples, rng)

    # KS test against the analytical HG CDF
    ks_stat, p_value = stats.kstest(
        theta_samples,
        lambda t: analytical_hg_cdf(t, g)
    )
    assert p_value > 0.05, (
        f"KS test failed for HG phase function (g={g}): "
        f"KS stat = {ks_stat:.4f}, p-value = {p_value:.4f}"
    )
```

**Expected result:** KS statistic < 0.005 for 100,000 samples; p-value > 0.05.

**Why this test matters:** The phase function sampler is the most likely source of systematic error. A bug in the CDF normalization, the sin(theta) weighting, or the interpolation would show up as a statistically significant KS deviation even with a few thousand samples.

---

## Test 3: Receiver Geometry (Lambertian Power Budget)

**Physical basis:** A Lambertian point source emits power proportional to cos(theta) per unit solid angle. For an isotropically emitting (in 2*pi steradians) source at the origin, the fraction of power intercepted by a circular aperture of radius R at distance D along the normal is:

```
F = 1 - cos(arctan(R/D)) = 1 - D / sqrt(D^2 + R^2)
```

(This is the exact view-factor formula for a disk and a point source.)

**Setup:** Use an isotropic point source (uniform direction cosines over the forward hemisphere), no medium attenuation (a = b = 0), a large receiver at z = D.

```python
# tests/test_receiver_geometry.py
import numpy as np
from photonator import Simulation, Receiver
from photonator.beams import IsotropicPointBeam   # uniform over forward hemisphere
from photonator.media import VacuumMedium         # a = b = 0

def test_receiver_power_budget():
    D_m = 1.0         # receiver distance (m)
    R_m = 0.5         # receiver radius (m)
    expected_fraction = 1 - D_m / np.sqrt(D_m**2 + R_m**2)   # ≈ 0.2469

    beam = IsotropicPointBeam()
    medium = VacuumMedium()
    rx = Receiver(
        position_xy=(0.0, 0.0), aperture_m=2*R_m,
        fov_rad=np.pi, fov_weighting='uniform', n_water=1.0, n_air=1.0
    )
    sim = Simulation(
        beam=beam, medium=medium, receiver=rx,
        receiver_z_m=D_m, n_photons=500_000, n_batches=4, seed=7
    )
    result = sim.run()

    measured_fraction = result.mean_weight
    assert abs(measured_fraction - expected_fraction) / expected_fraction < 0.02, (
        f"Receiver geometry error: expected {expected_fraction:.6f}, "
        f"got {measured_fraction:.6f} ({abs(measured_fraction-expected_fraction)/expected_fraction*100:.2f}%)"
    )
```

**Expected value:** View factor approximately 0.2469.

**Tolerance:** Relative error < 2%.

**Note:** With n_air = n_water = 1.0 (no interface), Fresnel transmission is T = 1. The test isolates purely geometric effects.

---

## Test 4: Layer Refraction (Snell's Law Angle)

**Physical basis:** A photon packet crossing a planar interface between media with refractive indices n1 and n2 must satisfy Snell's law:

```
n1 * sin(theta_i) = n2 * sin(theta_t)
```

**Setup:** Launch photons at a fixed angle theta_i = 30 degrees in water (n1 = 1.33) into air (n2 = 1.0). The transmitted angle should be:

```
theta_t = arcsin(n1/n2 * sin(theta_i)) = arcsin(1.33 * sin(30°)) = arcsin(0.665) ≈ 41.7°
```

```python
# tests/test_refraction.py
import numpy as np
from photonator import Simulation, Receiver
from photonator.beams import CollimatedBeam   # fixed angle, zero divergence
from photonator.media import VacuumMedium

def test_snell_law_angle():
    theta_i_rad = np.radians(30.0)
    n_water = 1.33
    n_air   = 1.0
    theta_t_expected = np.arcsin(n_water / n_air * np.sin(theta_i_rad))

    beam = CollimatedBeam(theta_polar_rad=theta_i_rad, phi_azimuth_rad=0.0)
    medium = VacuumMedium()
    rx = Receiver(
        position_xy=(0.0, 0.0), aperture_m=10.0, fov_rad=np.pi,
        fov_weighting='uniform', n_water=n_water, n_air=n_air
    )
    sim = Simulation(
        beam=beam, medium=medium, receiver=rx,
        receiver_z_m=1.0, n_photons=50_000, n_batches=1, seed=3
    )
    result = sim.run()

    # The mean uz after the interface corresponds to theta_t
    mean_uz = result.angle_mean   # uz in air after Fresnel
    theta_t_measured = np.arccos(mean_uz)
    error_deg = abs(np.degrees(theta_t_measured) - np.degrees(theta_t_expected))

    assert error_deg < 0.1, (
        f"Snell's law angle error: expected {np.degrees(theta_t_expected):.3f} deg, "
        f"measured {np.degrees(theta_t_measured):.3f} deg, error = {error_deg:.3f} deg"
    )
```

**Expected transmitted angle:** approximately 41.7 degrees.

**Tolerance:** Absolute angular error < 0.1 degrees.

**Note:** This test is essentially deterministic (no random scattering in the medium) so the MC variance is negligible. The error is limited by the resolution of the direction-cosine representation.

---

## Test 5: MATLAB Comparison

This test compares the Python simulation output against saved MATLAB output files generated by `photon_sim_r4.m` / `mc_func_r6.m` / `mc_rec_r5.m` running with identical physical parameters.

**Reference MATLAB run parameters (stored in simVariables.mat):**

```
c = 0.15 m^-1,  a = 0.025 m^-1,  b = 0.125 m^-1
albedo = 0.8333 (= b/c)
receiver_z = 16/0.15 = 106.67 m
num_photons = 1e5,  num_sims = 10 (1e6 total)
beamWidth = 0.001 m,  beamDiverg = 7.5e-4 rad
rec_aperture = 0.8 m,  rec_fov = pi/2 rad
VSF: petzold_clear (petzold_ocean.mat column 4)
```

**How to run the MATLAB comparison:**

```bash
python -m photonator.validation.matlab_compare \
    --mat-dir /path/to/matlab/output/           \
    --n-photons 1000000                          \
    --seed 0
```

The script:

1. Loads `simVariables.mat` and all per-batch result `.mat` files from the directory.
2. Extracts the concatenated `rec_weights` (received-photon weights) and `total_rec_dist` (path lengths) from the MATLAB output.
3. Runs the Python simulation with identical parameters.
4. Compares the following metrics:

| Metric | Tolerance | Method |
|--------|-----------|--------|
| Total received power normalized (sum of weights / N_launched) | 1% relative | Absolute value comparison |
| Mean arrival angle (mean of uz at receiver) | 0.5 degrees | Absolute angle difference |
| Path length distribution | KS test p > 0.05 | Kolmogorov-Smirnov two-sample test |
| Radial distribution at receiver | KS test p > 0.05 | KS two-sample test on sqrt(x^2+y^2) |

**Example output (passing):**

```
MATLAB comparison validation
=============================
MATLAB received power:   3.42e-03 (normalized)
Python received power:   3.39e-03 (normalized)
Relative error:          0.88%  [PASS, threshold 1%]

MATLAB mean angle:       2.14 degrees
Python mean angle:       2.11 degrees
Angular error:           0.03 degrees  [PASS, threshold 0.5 deg]

Path length KS test:     D=0.0082, p=0.312  [PASS, p > 0.05]
Radial KS test:          D=0.0091, p=0.274  [PASS, p > 0.05]
```

**Implementation (stub):**

```python
# photonator/validation/matlab_compare.py
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.stats import ks_2samp
from photonator import Simulation, Receiver
from photonator.beams import GaussianBeam
from photonator.media import HomogeneousMedium
from photonator.phase_functions import PetzoldPhaseFunction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-dir', required=True)
    parser.add_argument('--n-photons', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load MATLAB reference data
    sv = loadmat(f"{args.mat_dir}/simVariables.mat")
    c   = float(sv['c'])
    a   = float(sv['a'])
    b   = float(sv['b'])
    receiver_z = float(sv['receiver_z'])
    beam_width = float(sv['beamWidth'])
    beam_diverg = float(sv['beamDiverg'])
    rec_aperture = float(sv['rec_aperture'])
    rec_fov = float(sv['rec_fov'])
    n_total = int(sv['num_photons']) * int(sv['num_sims'])

    # Concatenate all batch results
    matlab_weights = []
    matlab_dists = []
    matlab_locs = []
    import glob, os
    for f in sorted(glob.glob(os.path.join(args.mat_dir, 'simOutput_*.mat'))):
        d = loadmat(f)
        matlab_weights.append(d['rec_weights'].ravel())
        matlab_dists.append(d['total_rec_dist'].ravel())
        matlab_locs.append(d['rec_loc_final'])
    matlab_weights = np.concatenate(matlab_weights)
    matlab_dists = np.concatenate(matlab_dists)
    matlab_locs = np.vstack(matlab_locs)

    matlab_power = matlab_weights.sum() / n_total
    matlab_angle = np.degrees(np.arccos(np.mean(matlab_locs[:, 4])))

    # Run Python simulation
    pf = PetzoldPhaseFunction.from_file('petzold_clear')
    beam = GaussianBeam(waist_m=beam_width, divergence_rad=beam_diverg)
    from photonator.media import HomogeneousMedium
    medium = HomogeneousMedium(a_per_m=a, b_per_m=b, phase_function=pf)
    rx = Receiver(
        position_xy=(0.0, 0.0),
        aperture_m=rec_aperture,
        fov_rad=rec_fov,
        fov_weighting='two_term_gaussian'
    )
    sim = Simulation(
        beam=beam, medium=medium, receiver=rx,
        receiver_z_m=receiver_z,
        n_photons=args.n_photons,
        n_batches=1,
        seed=args.seed
    )
    result = sim.run()

    python_power = result.mean_weight
    python_angle_deg = np.degrees(np.arccos(result.angle_mean))
    python_dists = result.path_lengths
    python_radii = result.radii

    # --- Comparisons ---
    power_err_pct = abs(python_power - matlab_power) / matlab_power * 100
    angle_err_deg = abs(python_angle_deg - matlab_angle)
    ks_path = ks_2samp(matlab_dists, python_dists)
    matlab_radii = np.sqrt(matlab_locs[:, 0]**2 + matlab_locs[:, 1]**2)
    ks_radial = ks_2samp(matlab_radii, python_radii)

    print(f"Relative power error:  {power_err_pct:.2f}%  {'PASS' if power_err_pct < 1 else 'FAIL'}")
    print(f"Angular error:         {angle_err_deg:.3f} deg  {'PASS' if angle_err_deg < 0.5 else 'FAIL'}")
    print(f"Path length KS p:      {ks_path.pvalue:.3f}  {'PASS' if ks_path.pvalue > 0.05 else 'FAIL'}")
    print(f"Radial KS p:           {ks_radial.pvalue:.3f}  {'PASS' if ks_radial.pvalue > 0.05 else 'FAIL'}")

if __name__ == '__main__':
    main()
```

---

## Test Coverage Summary

| Test | File | Validates |
|------|------|-----------|
| Beer-Lambert | `tests/test_beer_lambert.py` | Path sampling, weight update, propagation |
| HG phase function | `tests/test_phase_function.py` | CDF construction, inverse sampling |
| Receiver geometry | `tests/test_receiver_geometry.py` | Aperture, FOV, view factor |
| Snell's law | `tests/test_refraction.py` | Fresnel interface, direction cosine update |
| MATLAB comparison | `photonator/validation/matlab_compare.py` | Full end-to-end parity with MATLAB |

Run all standard tests with:

```bash
pytest tests/ -v --tb=short
```

For continuous integration, the MATLAB comparison test is skipped unless `PHOTONATOR_TEST_DATA` is set, since it requires reference `.mat` files not stored in the repository.
