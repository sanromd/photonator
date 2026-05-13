# Monte Carlo Numerical Simulation Reference

## 1. What is MC Photon Transport

Monte Carlo photon transport is a stochastic numerical method for solving the Radiative Transfer Equation (RTE):

```
(1/v)(dL/dt) + u_hat · grad(L) + c*L = b * integral[ p(u_hat, u_hat') * L(r, u_hat', t) dOmega' ] + S
```

where L is the radiance (W m^-2 sr^-1), c = a + b is the volume attenuation coefficient (m^-1), b is the scattering coefficient, p is the phase function (sr^-1), and S is an internal source term.

The RTE has no general closed-form solution for arbitrary geometry and phase functions. MC photon transport replaces the deterministic solution with an ensemble average over many independent photon packet histories. Each photon packet follows a random walk governed by probability distributions derived directly from the physical interaction cross-sections. By the law of large numbers, the ensemble average converges to the true radiance field as the number of packets N grows.

**Key properties:**

- **Unbiased:** The expected value of the MC estimator equals the true solution (there is no systematic error from spatial discretization, unlike finite-difference or finite-element solvers).
- **Variance decreases as 1/N:** The standard deviation of any observable scales as 1/sqrt(N), regardless of the dimensionality of the problem.
- **Geometry-agnostic:** Complex boundaries, non-uniform media, and arbitrary phase functions are handled by the same random-walk algorithm.
- **Computationally intensive:** Achieving 1% precision (2 significant figures) requires N approximately 10,000 detected events; 0.1% precision requires N approximately 1,000,000.

The photonator implementation follows Wang, Jacques, and Zheng (1995) "MCML" conventions adapted for aquatic rather than biological tissue media.

---

## 2. Photon Packet Weight

Rather than tracking individual photons that are either absorbed or not at each collision (analog MC), photonator uses **implicit capture** (also called the statistical weight or survival biasing technique).

Each photon packet carries a weight `w` in [0, 1], initially set to 1.0. At each scattering event, instead of gambling on whether the photon is absorbed, the weight is reduced by the survival probability:

```
w_new = w_old * (b / c) = w_old * omega_0
```

where `omega_0 = b/c` is the single-scattering albedo. This is equivalent to multiplying by the probability that the photon was scattered rather than absorbed at that interaction site.

The implicit-capture weight represents the fraction of the original photon packet's energy that has survived to the current position. The accumulated weight of photons reaching the detector is therefore an unbiased estimator of the radiometric power ratio (received power / transmitted power).

**Advantages over analog MC:**
- Every photon packet contributes to the statistics, even in highly absorbing media where most photons would be lost to absorption in analog MC.
- Variance is reduced for problems where omega_0 is close to 1 (weakly absorbing media, common in clear water at visible wavelengths).

The weight track is:
```
w_k = w_0 * omega_0^k
```
after k scattering events (ignoring roulette). This is the discrete form of the Beer-Lambert decay for the absorbed fraction.

---

## 3. Path Length Sampling

Between scattering events, photon packets travel in a straight line. The probability that a photon travels a distance r before its next interaction is given by the exponential distribution:

```
p(r) = c * exp(-c * r),   r >= 0
```

The cumulative distribution function (CDF) is:

```
F(r) = 1 - exp(-c * r)
```

Inverting: given a uniform random variate U ~ Uniform(0, 1):

```
r = -ln(U) / c = -(1/c) * ln(U)
```

This is **inverse transform sampling** applied to the exponential distribution. In the MATLAB code:

```matlab
r = -inv_c * log(rand_array(i,1))
```

where `inv_c = 1/c` is precomputed for efficiency.

**Physical interpretation:** The mean free path (average distance between interactions) is 1/c meters. For clear ocean water at 532 nm, c approximately 0.15 m^-1, giving a mean free path of approximately 6.7 m. For harbor water, c approximately 2.19 m^-1, giving a mean free path of approximately 0.46 m.

**Note:** The path length is sampled using the total attenuation coefficient c = a + b, not just the scattering coefficient b. This correctly accounts for both absorption and scattering events. The absorption is then handled implicitly through weight reduction (see Section 2).

---

## 4. Phase Function and CDF Sampling

### Definition and normalization

The phase function p(theta) (sr^-1) describes the probability density for a photon to scatter into direction theta relative to its current propagation direction. It must satisfy the normalization condition over the full sphere:

```
integral over sphere [ p(theta) dOmega ] = 1

(1/2) * integral_0^pi [ p(theta) * sin(theta) ] dtheta = 1
```

The factor sin(theta) comes from the solid-angle element `dOmega = sin(theta) dtheta dphi`, integrated over phi from 0 to 2*pi (which gives a factor of 2*pi) and divided by 4*pi (the total solid angle of the sphere), yielding the sin(theta)/2 prefactor.

### Building the CDF

Given a tabulated VSF beta(theta) (m^-1 sr^-1), the normalized phase function is p(theta) = beta(theta) / b where b = integral[beta(theta) dOmega]. In practice, the CDF is built as:

```python
# In MATLAB:
cdf = cumtrapz(angle_rad, VSF * sin(angle_rad))
cdf = cdf / cdf[-1]   # normalize to [0, 1]

# In Python (equivalent):
cdf = np.cumulative_trapezoid(vsf * np.sin(angle_rad), angle_rad)
cdf /= cdf[-1]
```

The sin(theta) factor inside the integral is essential. Without it, the CDF would weight all angles equally regardless of the solid angle they represent. With it, the CDF correctly accounts for the fact that near-forward (small theta) and near-backward (theta near pi) angles subtend a tiny solid angle and should contribute less to the integral even if the VSF value there is large.

### Inverse transform sampling

To draw a scattering angle from p(theta), generate U ~ Uniform(0, 1) and find theta* such that CDF(theta*) = U. In the MATLAB code this is done by binary search followed by linear interpolation:

```matlab
% Binary search (lookupCDF) finds bracket index k
theta = angle(k-1) + (U - cdf(k-1)) * (angle(k) - angle(k-1)) / (cdf(k) - cdf(k-1))
```

In Python:

```python
k = np.searchsorted(cdf, u_array)     # vectorized, O(N log M)
theta = np.interp(u_array, cdf, angle_rad)   # linear interpolation
```

### Why the sin(theta) factor appears in cumtrapz

Physically: the probability that a photon scatters into an annular ring between theta and theta+dtheta is proportional to p(theta) times the solid angle of that ring, which is 2*pi*sin(theta)*dtheta. So the 1D marginal PDF over theta alone is:

```
f(theta) = (1/2) * p(theta) * sin(theta)
```

Integrating this over [0, pi] gives 1 (by the normalization of p). The CDF built from VSF*sin(theta) is therefore the CDF of this marginal distribution, giving exactly the correct sampling of the polar scattering angle.

---

## 5. Direction Cosine Update

After sampling the polar scattering angle theta and the azimuthal angle phi (uniform on [0, 2*pi]), the direction cosine vector (ux, uy, uz) must be rotated. The standard rotation used in mc_func_r6 is:

**General case** (|uz| < 1 - 1e-12):

```
sqrt_uz = sqrt(1 - uz^2)

ux_new = (sin(theta) / sqrt_uz) * (ux*uz*cos(phi) - uy*sin(phi)) + ux*cos(theta)
uy_new = (sin(theta) / sqrt_uz) * (uy*uz*cos(phi) + ux*sin(phi)) + uy*cos(theta)
uz_new = -sin(theta)*cos(phi)*sqrt_uz + uz*cos(theta)
```

**Degenerate case** (|uz| >= 1 - 1e-12, beam nearly axial):

```
ux_new = sin(theta) * cos(phi)
uy_new = sin(theta) * sin(phi)
uz_new = sign(uz) * cos(theta)
```

This rotation is derived by composing two rotations: first align the z-axis with the current propagation direction (ux, uy, uz), then rotate by (theta, phi) in the local frame, then transform back to the lab frame. The formula is algebraically equivalent to the Rodrigues rotation formula for the special case of rotating from the z-axis.

**Normalization check:** After the update, if the magnitude of the new direction vector deviates from unity by more than 1e-11, the vector is explicitly renormalized:

```matlab
normLength = sqrt(ux^2 + uy^2 + uz^2)
ux = ux / normLength
uy = uy / normLength
uz = uz / normLength
```

This guard prevents accumulated floating-point errors from causing the direction vector to drift off the unit sphere over many scattering events.

---

## 6. Roulette Variance Reduction

As photons scatter and lose weight through implicit absorption, eventually their weight drops below a threshold `min_power` below which their contribution to the detected signal is negligible. Simply discarding them would introduce a negative bias. Roulette is the standard unbiased technique for terminating low-weight photons.

**Algorithm:**

1. After each scatter event, if `w < min_power`:
2. Draw a uniform variate U.
3. If `U > 1/rouletteConst` (probability = `(rouletteConst - 1)/rouletteConst`): terminate the photon (`status = -1`).
4. Otherwise (probability = `1/rouletteConst`): multiply the weight by `rouletteConst` and continue.

In the MATLAB code, `rouletteConst = 10`, so each surviving photon has its weight multiplied by 10 and the termination probability is 90%.

**Unbiasedness:** The expected weight change is:
```
E[w_after] = (1/10) * (10*w) + (9/10) * 0 = w
```

The expected value is preserved, so the estimator remains unbiased. Roulette introduces additional variance (it is not a variance-reduction technique for the weight itself) but it prevents the simulation from being dominated by infinitely many vanishingly small weights.

**Threshold selection:** The code adapts `min_power` based on the albedo:
- omega_0 >= 0.90: `min_power = 1e-4`
- omega_0 >= 0.83: `min_power = 1e-5`
- omega_0 >= 0.70: `min_power = 1e-5`
- omega_0 < 0.70:  `min_power = 1e-7`

Highly absorbing media (low albedo) need a lower threshold because weights decay faster and photons with weight 1e-4 may still represent significant energy relative to the noise floor.

---

## 7. Receiver Detection

Photons that reach the receiver plane (z >= receiver_z) pass through a multi-stage detection filter in mc_rec_r5:

### Stage 1: Critical angle test

The receiver is immersed in water (n_water = 1.33); the detector sits in air (n_air = 1.0). Total internal reflection occurs for photons arriving at angles beyond the critical angle:

```
theta_c = arcsin(n_air / n_water) = arcsin(1/1.33) approximately 48.8 degrees
cos(theta_c) = sqrt(1 - (n_air/n_water)^2) approximately 0.6614
```

Photons with `uz < cos(theta_c)` (i.e., arriving at more than 48.8 degrees from normal) are rejected.

### Stage 2: Fresnel transmission

For photons that pass Stage 1, Fresnel transmission coefficients are computed:

```
cosExitAng = sqrt(1 - (n_water/n_air)^2 * sin^2(theta_incident))
           = sqrt(1 - (n_water/n_air)^2 * (1 - uz^2))

rp = (uz - n_water * cosExitAng) / (uz + n_water * cosExitAng)
rs = (cosExitAng - n_water * uz) / (cosExitAng + n_water * uz)
R  = (rp^2 + rs^2) / 2    (unpolarized light)
T  = 1 - R
```

The photon weight is multiplied by T. The direction cosine uz is updated to cosExitAng (the transmitted ray direction in air).

### Stage 3: Aperture test

The Euclidean distance from the photon's arrival position (ph_x, ph_y) to the receiver center (rx_x, rx_y) is computed. The photon is accepted only if `distance <= rec_aperture/2`.

### Stage 4: FOV test

The photon's polar angle from the receiver's boresight must satisfy `uz >= cos(rec_fov/2)`.

### Stage 5: Two-term Gaussian angular weighting

The receiver's angular sensitivity is modeled as a two-term Gaussian in arrival angle theta = arccos(uz):

```
fovWeight = A1*exp(-((theta - B1)/C1)^2) + A2*exp(-((theta - B2)/C2)^2)
```

with A1=0.7985, B1=0.0187, C1=0.03437, A2=0.7121, B2=-0.02337, C2=0.03117. This models the measured angular response of the physical detector used in the laboratory experiment.

### Welford online statistics

Running mean and variance are computed for arrival angle, path length, and weighted power using Welford's one-pass algorithm, which is numerically stable and does not require storing all samples:

```
n = n + 1
delta = x - mean
mean  = mean + delta / n
mean2 = mean2 + delta * (x - mean)   # Note: updated mean used here
```

At the end, `variance = mean2 / (n - 1)` (Bessel-corrected sample variance).

---

## 8. Convergence

### How many photons to simulate

The variance of the estimated received power scales as:

```
Var[P_hat] proportional to 1/N
std[P_hat] proportional to 1/sqrt(N)
```

As a rule of thumb:

| Desired precision | Required detected events |
|-------------------|--------------------------|
| 10% (1 sig. fig.) | ~100 |
| 3% | ~1,000 |
| 1% | ~10,000 |
| 0.3% | ~100,000 |
| 0.1% (3 sig. fig.) | ~1,000,000 |

"Detected events" means photon packets that actually reach the receiver after passing all detection filters, not total launched packets. For deep-water scenarios (many attenuation lengths) the detection fraction can be as low as 10^-6, requiring 10^8 or more launched photons to accumulate 100 detected events.

### Choosing num_photons

The default setting of `num_photons = 1e5` per batch and `num_sims = 10` (1 million total) is appropriate for:
- Moderate attenuation depths (1–16 optical lengths)
- Wide-FOV receivers (FOV > 30 degrees)
- Clear or coastal water

For harbor water, narrow-FOV receivers, or many attenuation lengths, increase `num_photons` or `num_sims` by 1–2 orders of magnitude. Monitor convergence by comparing successive batches: if the mean power changes by more than the target precision, add more photons.

### Variance estimate

The Welford accumulator in mc_rec_r5 provides the sample variance of received-photon weights. The standard error of the mean estimated power is:

```
SEM = sqrt(weight_var / num_detected)
```

A relative uncertainty below 1% indicates adequate sampling for most applications.

---

## 9. Workflow Diagram

```
 ┌─────────────────────────────────────────────────────────────┐
 │                   photon_sim_r4 (orchestrator)              │
 │  Set: c, a, b, omega_0, receiver geometry, beam params      │
 │  Build CDF: generate_scatter → cumtrapz(VSF*sin(theta))     │
 └───────────────────┬─────────────────────────────────────────┘
                     │ for each batch (num_sims)
                     v
 ┌─────────────────────────────────────────────────────────────┐
 │              mc_func_r6 (core MC loop)                      │
 │                                                             │
 │  Initialize                                                 │
 │    beamProfile → (x, y, ux, uy, uz), w=1, status=active    │
 │       │                                                     │
 │       v                                                     │
 │  while photonsRemaining > 0:                                │
 │    Sample r = -(1/c)*ln(U)   ← path length                 │
 │       │                                                     │
 │       v                                                     │
 │    Move photon: (x,y,z) += r*(ux,uy,uz)                    │
 │       │                                                     │
 │       ├─── z >= receiver_z? ──YES──► Ray-trace to plane     │
 │       │                               Record (x,y,ux,uy,uz)│
 │       │                               status = received     │
 │       │                               photonsRemaining -= 1 │
 │       │                                                     │
 │       NO                                                    │
 │       │                                                     │
 │       v                                                     │
 │    Apply implicit absorption: w *= omega_0                  │
 │       │                                                     │
 │       v                                                     │
 │    Roulette: w < min_power?                                 │
 │       ├─ YES, U > 1/10 ──► terminate (photonsRemaining-=1)  │
 │       ├─ YES, U <= 1/10 ──► w *= 10, continue              │
 │       └─ NO ──────────────► continue                       │
 │       │                                                     │
 │       v                                                     │
 │    Sample scatter angle: theta = inv-CDF(U)                 │
 │    Sample phi = 2*pi*U                                      │
 │    Rotate direction cosines (ux, uy, uz) → (ux', uy', uz') │
 │    Renormalize if needed                                     │
 │       │                                                     │
 │       └────────────────────────────────── loop ─────────────┘
 │                                                             │
 └───────────────────┬─────────────────────────────────────────┘
                     │ rec_loc_final, rec_weights, total_rec_dist
                     v
 ┌─────────────────────────────────────────────────────────────┐
 │              mc_rec_r5 (receiver)                           │
 │                                                             │
 │  For each photon at receiver plane:                         │
 │    1. Critical angle test: uz < cos(theta_c)? → skip        │
 │    2. Fresnel transmission: w *= T(uz)                      │
 │    3. Aperture test: distance > radius? → skip              │
 │    4. FOV test: uz < cos(fov/2)? → skip                    │
 │    5. Angular weighting: w *= fovWeight(theta)              │
 │    6. Accumulate: power, ph_cnt                             │
 │    7. Welford update: mean/var of angle, dist, weight       │
 │                                                             │
 │  Output: power, ph_cnt, angle stats, dist stats, weight var │
 └─────────────────────────────────────────────────────────────┘
```

---

## 10. Extension Hooks

The following locations in the physics model are natural extension points for additional capabilities:

### Wideband (spectral) sources

**Where to plug in:** The Medium class (Python). Instead of a single scalar `c`, `a`, `b`, maintain arrays indexed by wavelength. The photon packet gains a wavelength attribute (or a wavelength index for a discrete spectral grid). At each step, look up the medium properties for the photon's wavelength.

**Beam:** The AbstractBeam initializer samples wavelength from the source spectral distribution and assigns it to each photon packet.

**Phase function:** AbstractPhaseFunction.sample_angle() receives the photon's wavelength and returns a CDF appropriate for that wavelength.

**Aggregation:** Received-power contributions are binned by wavelength to produce a spectral irradiance at the detector.

### Inelastic scattering (fluorescence, Raman)

**Where to plug in:** AbstractPhaseFunction (Python), or a separate AbstractInelasticProcess class.

**Stub location in mc_func_r6 equivalent:** After the weight update step (`w *= omega_0`), before the scatter angle sample. If the medium has a non-zero inelastic yield Y(lambda_excitation), draw a variate to decide if this interaction is elastic or inelastic. If inelastic:
- Draw the emission wavelength from the fluorescence/Raman spectral profile.
- Update the photon's wavelength attribute.
- Sample the emission direction from the appropriate (isotropic for Raman, nearly isotropic for fluorescence) phase function.
- Reduce weight by the quantum yield.

**AbstractMedium stub:**
```python
def inelastic_yield(self, wavelength_nm: float) -> float:
    """Return probability of inelastic conversion per interaction.
    Override in subclasses for fluorescence/Raman media."""
    return 0.0

def emission_wavelength_nm(self, excitation_nm: float) -> float:
    """Sample emission wavelength for an inelastic event.
    Override with Stokes shift distribution."""
    raise NotImplementedError
```

### Additional beam profiles

Implement AbstractBeam.initialize(n_photons) to return (x, y, z, ux, uy, uz, weight) arrays. Existing implementations: GaussianBeam. Planned: FlatTopBeam, BesselBeam, LaguerreGaussianBeam, UserDefinedBeam(from_file).

### Additional phase functions

Implement AbstractPhaseFunction.sample_angle(n) to return theta array. Existing: PetzoldPhaseFunction, HenyeyGreensteinPhaseFunction. Planned: MiePhaseFunction (computed from Bohren-Huffman), TwoTermHGPhaseFunction, UserDefinedPhaseFunction(from_file).
