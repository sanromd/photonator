# Physics Reference

## Beer-Lambert Law and Attenuation Length

When a collimated beam of monochromatic light propagates through a homogeneous medium, the intensity decays exponentially with distance:

```
I(z) = I_0 * exp(-c * z)
```

where:
- `I_0` is the initial irradiance (W m^-2)
- `z` is the path length (m)
- `c` is the volume attenuation coefficient (m^-1)

The **attenuation length** (also called the extinction length or photon mean free path) is:

```
L_att = 1 / c   (m)
```

After one attenuation length, the beam intensity falls to 1/e approximately 36.8% of its initial value. After N attenuation lengths, the transmitted fraction is exp(-N).

The exponential decay arises because each infinitesimal volume element removes a constant fraction of the beam, leading to the differential equation dI/dz = -c*I.

---

## Absorption vs. Scattering Coefficients, Albedo

The total attenuation coefficient c is the sum of independent contributions from absorption and scattering:

```
c = a + b   (m^-1)
```

where:
- `a` is the absorption coefficient (m^-1): energy converted irreversibly to heat or fluorescence
- `b` is the scattering coefficient (m^-1): energy redirected without loss (elastic scattering)

The **single-scattering albedo** (or just albedo) is the fraction of attenuated photons that were scattered rather than absorbed:

```
omega_0 = b / c = b / (a + b)
```

Albedo ranges from 0 (pure absorber, no scattering) to 1 (pure scatterer, no absorption). Typical values:

| Water type | a (m^-1) | b (m^-1) | c (m^-1) | omega_0 |
|------------|----------|----------|----------|---------|
| Clear ocean (532 nm) | 0.0374 | 0.114 | 0.151 | 0.756 |
| Coastal | 0.179 | 0.220 | 0.399 | 0.551 |
| Harbor | 0.366 | 1.824 | 2.190 | 0.833 |
| Maalox suspension | ~0.005 | ~0.5–2 | varies | ~0.99 |

---

## Phase Function Normalization

The phase function p(theta) describes the angular probability distribution of scattered light. For a rotationally symmetric medium (azimuthal symmetry), p depends only on the polar scattering angle theta relative to the incident direction.

**Normalization condition over the full sphere:**

```
integral over 4*pi steradians [ p(theta) dOmega ] = 1

integral_0^pi integral_0^{2*pi} p(theta) * sin(theta) dphi dtheta = 1

2*pi * integral_0^pi p(theta) * sin(theta) dtheta = 1

(1/2) * integral_0^pi p(theta) * sin(theta) dtheta = 1
```

The sin(theta) factor is the Jacobian from spherical coordinates: the solid angle element is `dOmega = sin(theta) dtheta dphi`. Without the sin(theta) weighting, an isotropic distribution would incorrectly assign equal probability to near-axial and equatorial angles despite the latter subtending far more solid angle.

The volume-scattering function (VSF) beta(theta) in units of m^-1 sr^-1 is related to the phase function by:

```
p(theta) = beta(theta) / b
```

where `b = 2*pi * integral_0^pi beta(theta) * sin(theta) dtheta`.

---

## Henyey-Greenstein Phase Function

The Henyey-Greenstein (HG) phase function is a single-parameter analytic model widely used for biological tissue and ocean optics:

```
p_HG(theta; g) = (1/(4*pi)) * (1 - g^2) / (1 + g^2 - 2*g*cos(theta))^(3/2)
```

The asymmetry parameter g is the mean cosine of the scattering angle:

```
g = <cos(theta)> = (1/2) * integral_0^pi p(theta) * cos(theta) * sin(theta) dtheta
```

**Special cases:**
- g = 0: isotropic scattering (spherically symmetric)
- g = 1: completely forward-peaked (no actual scattering, delta function at theta=0)
- g = -1: completely backward-peaked (delta function at theta=pi)
- g = 0.93 (typical ocean): strongly forward-peaked, most photons scatter within a few degrees

**CDF for inverse-transform sampling (exact, closed-form):**

```
cos(theta) = (1/(2g)) * [1 + g^2 - ((1-g^2) / (1 - g + 2*g*U))^2]   for g != 0
cos(theta) = 1 - 2*U                                                   for g = 0
```

where U ~ Uniform(0, 1). This closed-form CDF makes HG the most computationally efficient phase function for MC simulations.

---

## Petzold / Haltrin Empirical Formula

Haltrin (1997) fitted an empirical formula to the Petzold (1972) VSF measurements for various ocean water types. The phase function is expressed as a polynomial-exponential in the scattering angle theta (in degrees):

```
p(theta) = (4*pi/b) * exp(q * (1 - k1*theta^(1/2) + k2*theta - k3*theta^(3/2) + k4*theta^2 - k5*theta^(5/2)))
```

The coefficients depend on the scattering coefficient b and albedo omega_0:

```
q  = 2.598 + 17.748*sqrt(b) - 16.722*b + 5.932*b*sqrt(b)
k1 = 1.188 - 0.688*omega_0
k2 = 0.1*(3.07 - 1.90*omega_0)
k3 = 0.01*(4.58 - 3.02*omega_0)
k4 = 0.001*(3.24 - 2.25*omega_0)
k5 = 0.0001*(0.84 - 0.61*omega_0)
```

This formula is valid for the Petzold measurement set (harbor, coastal, and clear ocean water). The exponent is a degree-5 polynomial in sqrt(theta), which captures the steep forward peak and the relatively flat backward hemisphere characteristic of aquatic scatterers.

**Reference:** Haltrin, V.I. (1997). "Theoretical and empirical phase functions for Monte Carlo calculations of light scattering in seawater." *Proceedings SPIE 3222*, Ocean Optics XIII.

---

## Mie Scattering (Reference)

Mie theory provides an exact solution to Maxwell's equations for electromagnetic scattering by a homogeneous sphere. For a sphere of radius r and complex refractive index m = n + ik in a medium of index n_medium:

- The scattering efficiency Q_sca and absorption efficiency Q_abs are computed from the Mie coefficients a_n and b_n (series in Riccati-Bessel functions).
- The differential scattering cross section yields the VSF directly.
- The size parameter x = 2*pi*r*n_medium/lambda determines the scattering regime.

For aquatic particles (phytoplankton, minerals, bacteria) spanning r approximately 0.1–100 microns and visible wavelengths, x ranges from 0.3 to 3000, well within the Mie regime. The resulting VSFs are strongly forward-peaked with g typically 0.85–0.99.

The photonator data files include Mie VSF for 1-micron spheres (mieVSF1micron in `Berrocal/mieVSF1micron.mat`) as a reference case. Full Mie calculations are not implemented in the current code; the `generate_scatter.m` file loads precomputed VSF tables.

**Recommended references:** Bohren, C.F. and Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley. The `miepython` Python package provides efficient Mie calculations suitable for building VSF tables for custom particle populations.

---

## Fresnel Equations

At a planar interface between two media with refractive indices n1 (incident) and n2 (transmitted), the amplitude reflection coefficients for s-polarized (perpendicular) and p-polarized (parallel) light are:

```
r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))
r_p = (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t))
```

where theta_i is the angle of incidence (from normal) and theta_t is the angle of transmission (from Snell's law).

The **reflectance** (intensity reflection coefficient) for unpolarized light is:

```
R = (r_s^2 + r_p^2) / 2
```

The **transmittance** is:

```
T = 1 - R
```

Note: T + R = 1 (energy conservation), but this assumes the transmitted and reflected beams have the same cross-sectional area, which is only true for plane waves. For beams with finite cross-section, a geometric factor corrects for the beam-width change at the interface, but this correction is not applied in the current code (paraxial approximation).

**As implemented in mc_rec_r5 for the water-to-air interface** (n1 = n_water = 1.33, n2 = n_air = 1.0):

```
cosExitAng = sqrt(1 - (n_water/n_air)^2 * (1 - uz^2))

rp = (uz - n_water*cosExitAng) / (uz + n_water*cosExitAng)
rs = (cosExitAng - n_water*uz) / (cosExitAng + n_water*uz)
R  = (rp^2 + rs^2) / 2
T  = 1 - R
```

Here `uz = cos(theta_i)` and `cosExitAng = cos(theta_t)`.

---

## Snell's Law

At an interface between two media, the angle of transmission is related to the angle of incidence by:

```
n1 * sin(theta_i) = n2 * sin(theta_t)
```

In terms of direction cosines (uz = cos(theta_i)):

```
sin(theta_t) = (n1/n2) * sin(theta_i) = (n1/n2) * sqrt(1 - uz^2)
cos(theta_t) = sqrt(1 - (n1/n2)^2 * (1 - uz^2))
```

The transverse components of the wave vector are preserved across the interface. In the code, the transmitted uz is computed directly as `cosExitAng = sqrt(1 - (n_water/n_air)^2 * (1 - uz^2))`.

---

## Total Internal Reflection Critical Angle

Total internal reflection (TIR) occurs when light travels from a denser medium (n1 > n2) and the angle of incidence exceeds the critical angle:

```
sin(theta_c) = n2 / n1
theta_c = arcsin(n2 / n1)
```

For the water-to-air interface (n_water = 1.33, n_air = 1.0):

```
theta_c = arcsin(1/1.33) approximately 48.75 degrees
cos(theta_c) = sqrt(1 - (1/1.33)^2) approximately 0.6614
```

In the code, photons with `uz < cos(theta_c)` (incident angle greater than theta_c from normal) are rejected as totally internally reflected. This is the condition `mu_z <= critAngCos` in mc_rec_r5 and the reflection loop in mc_func_r6.

---

## Direction Cosine Rotation Matrix

The scattering event rotates the photon direction by polar angle theta (scattering angle) and azimuthal angle phi (uniformly random). The rotation is expressed in direction-cosine form to avoid gimbal-lock and to operate directly on (ux, uy, uz) without converting to angles.

The update formula (from Prahl, Wang, Jacques 1993, and MCML code) for the general case |uz| < 1 - epsilon:

```
s = sqrt(1 - uz^2)    (sin of polar angle of current direction)

ux_new = (sin(theta)/s) * (ux*uz*cos(phi) - uy*sin(phi)) + ux*cos(theta)
uy_new = (sin(theta)/s) * (uy*uz*cos(phi) + ux*sin(phi)) + uy*cos(theta)
uz_new = -sin(theta)*cos(phi)*s + uz*cos(theta)
```

For the degenerate case |uz| >= 1 - epsilon (photon traveling nearly along the z-axis):

```
ux_new = sin(theta) * cos(phi)
uy_new = sin(theta) * sin(phi)
uz_new = sign(uz) * cos(theta)
```

This formula can be derived as follows. Define the local coordinate system with e_z aligned with the current direction (ux, uy, uz). The rotated direction in lab coordinates is obtained by multiplying the local scattering direction (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) by the rotation matrix that maps e_z to (ux, uy, uz). The result is the formula above.

---

## Gaussian Beam

### Irradiance profile

The transverse irradiance of a Gaussian beam at its waist (z = 0) is:

```
I(r) = I_0 * exp(-2*r^2 / w_0^2)
```

where `w_0` is the 1/e^2 intensity radius (beam waist) and r is the transverse radius. The factor of 2 in the exponent ensures that the 1/e^2 intensity points are at r = w_0.

### CDF for radius sampling

The probability that a random photon from this beam has transverse radius less than r is:

```
F(r) = 1 - exp(-r^2 / w_0^2)
```

Inverting: given U ~ Uniform(0, 1):

```
r = w_0 * sqrt(-log(1 - U))
```

This is the formula used in beamProfile.m. Note the absence of the factor of 2 from the irradiance formula: it is absorbed into the definition of w_0 used as the sampling parameter (which equals the 1/sqrt(e) intensity radius, not the 1/e^2 radius).

### Thin-lens divergence model

A thin lens with focal length f = -w_0/diverg (negative, a diverging lens) maps the position at the beam waist to a divergence angle:

```
theta_div(r) = r * diverg / w_0 = (diverg/w_0) * r
```

The marginal ray at r = w_0 has divergence angle exactly equal to `diverg`. This models the physical situation where the beam exits a focusing optic that sets the far-field divergence to match the diffraction limit (or an engineered value for a non-ideal beam).

Direction cosines after applying divergence:

```
uz = cos(theta_div(r))
ux = sin(theta_div(r)) * cos(phi)
uy = sin(theta_div(r)) * sin(phi)
```

---

## Laguerre-Gaussian Beam Profiles

Laguerre-Gaussian (LG) beams are higher-order solutions to the paraxial wave equation, characterized by radial index p and azimuthal index l (orbital angular momentum). The irradiance profile is:

```
I_{p,l}(r, phi) = I_0 * (r*sqrt(2)/w_0)^(2|l|) * [L_p^|l|(2r^2/w_0^2)]^2 * exp(-2r^2/w_0^2)
```

where `L_p^|l|` is the associated Laguerre polynomial. The l=0, p=0 mode is the fundamental Gaussian. For l != 0, the beam has a phase singularity (optical vortex) at r=0 and forms a ring-shaped intensity pattern.

**Sampling:** The radial CDF for LG beams does not have a closed-form inverse. Sampling is done by:
1. Compute the normalized CDF numerically on a fine radial grid.
2. Sample from the CDF using np.searchsorted and linear interpolation.
3. The azimuthal coordinate phi is uniform on [0, 2*pi] for all LG modes.

**Extension stub:** Implement `LaguerreGaussianBeam(AbstractBeam)` with parameters `l` (topological charge) and `p` (radial index). The phase structure (helical phase front) does not affect intensity-only MC transport but is relevant for coherence-sensitive calculations.

---

## Bessel Beam Profiles

A Bessel beam has a transverse irradiance profile given by a Bessel function of the first kind:

```
I(r) = I_0 * J_0(k_r * r)^2
```

where `k_r = k * sin(theta_cone)` is the transverse wave number and theta_cone is the cone half-angle of the generating annular aperture. Ideal Bessel beams are non-diffracting and have infinite energy; practical approximations are truncated by an aperture or Gaussian envelope (Bessel-Gauss beams).

For MC purposes, the intensity distribution is sampled numerically. The oscillating ring structure of J_0^2 means the CDF is monotone but has regions of slow growth (between rings), requiring a fine tabulation grid.

---

## Airy Beam Profiles

An Airy beam propagates along a curved trajectory in free space (self-accelerating). The transverse profile along one axis is:

```
I(x) proportional to Ai(x/x_0)^2
```

where `Ai` is the Airy function and x_0 is a transverse scale. The intensity peaks near the main lobe and decays with oscillating side lobes for negative argument.

Airy beams are inherently 2D asymmetric (different profiles in x and y). For MC initialization, sample x and y independently from the 1D marginal distributions derived from Ai^2, or use rejection sampling from a 2D grid.

---

## Extension Stubs: Spectral Dependence

To support wideband sources, the AbstractMedium interface reserves the following methods:

```python
def absorption_coeff(self, wavelength_nm: float) -> float:
    """Return absorption coefficient a (1/m) at the given wavelength.
    Default implementation: constant value from constructor."""
    return self._a

def scattering_coeff(self, wavelength_nm: float) -> float:
    """Return scattering coefficient b (1/m) at the given wavelength."""
    return self._b

def phase_function(self, wavelength_nm: float) -> AbstractPhaseFunction:
    """Return the phase function object for the given wavelength."""
    return self._phase_function
```

Per-wavelength lookup tables for a, b, and g can be loaded from the materials database (see materials.md) and interpolated to any wavelength in the source spectrum. The simulation loop then draws a wavelength from the source spectral distribution at initialization and uses the wavelength-specific properties throughout each photon's history.

## Extension Stubs: Inelastic Yield

```python
def inelastic_yield(self, wavelength_nm: float) -> float:
    """Probability of inelastic scattering per interaction (0 to 1).
    Default: 0.0 (elastic only). Override for fluorescent or Raman media."""
    return 0.0

def emission_wavelength_nm(self, excitation_nm: float) -> float:
    """Sample emission wavelength for an inelastic event.
    Must be overridden if inelastic_yield > 0."""
    raise NotImplementedError("Inelastic emission not implemented for this medium.")
```
