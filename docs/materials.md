# Materials Reference

This document describes the optical material models available in photonator, including the data sources, parameter ranges, and how to load custom materials. All coefficients are in SI units (m^-1 for attenuation, m^-1 sr^-1 for VSF) unless noted otherwise.

---

## Water Model

### Absorption: Pope and Fry (1997)

The most accurate published absorption spectrum for pure liquid water in the visible range is from Pope and Fry (1997), measured with a long-path integrating-cavity technique. Selected values at visible wavelengths:

| Wavelength (nm) | a_water (m^-1) |
|-----------------|----------------|
| 400 | 0.00663 |
| 450 | 0.00385 |
| 488 | 0.00468 |
| 532 | 0.0443 |
| 550 | 0.0596 |
| 589 | 0.0707 |
| 600 | 0.2436 |
| 650 | 0.3500 |
| 700 | 0.6395 |

At 532 nm (Nd:YAG second harmonic, most common underwater laser wavelength), `a_water = 0.0443 m^-1`. This is the value used for "pure water" in the photonator clear-ocean model.

**Reference:** Pope, R.M. and Fry, E.S. (1997). "Absorption spectrum (380–700 nm) of pure water. II. Integrating cavity measurements." *Applied Optics* 36(33), 8710–8723.

### Scattering: Petzold VSF

The Petzold (1972) measurements provide the most widely used empirical VSF data for ocean water. The `petzold_ocean.mat` file contains three water types:

| Water type | b (m^-1) | c (m^-1) | omega_0 | Measurement location |
|------------|----------|----------|---------|----------------------|
| Harbor | 1.824 | 2.190 | 0.833 | San Diego Harbor |
| Coastal | 0.220 | 0.399 | 0.551 | Pacific coastal |
| Clear ocean | 0.114 | 0.151 | 0.756 | Pacific clear |

**Turbidity scaling:** To model water with a different scattering coefficient b' while retaining the Petzold VSF shape, multiply the VSF by the ratio b'/b_petzold. The phase function shape (and hence g) is preserved; only the magnitude changes.

```python
from photonator.phase_functions import PetzoldPhaseFunction

# Use Petzold clear-ocean shape but with b = 0.3 m^-1 instead of 0.114
pf = PetzoldPhaseFunction.from_preset('petzold_clear', scale_b_to=0.3)
```

**Reference:** Petzold, T.J. (1972). "Volume scattering functions for selected ocean waters." SIO Reference 72-78. Scripps Institution of Oceanography, San Diego.

---

## Oil Model

### Literature IOR ranges

Crude and refined petroleum oils have a refractive index n that depends on composition, temperature, and wavelength. Representative values at 20°C and 589 nm (sodium D line):

| Oil type | n (real) | k (imaginary at 532 nm) |
|----------|----------|------------------------|
| Light crude | 1.46–1.48 | < 0.01 |
| Heavy crude | 1.48–1.52 | 0.01–0.1 |
| Refined (diesel) | 1.44–1.46 | < 0.001 |
| Refined (gasoline) | 1.43–1.45 | < 0.001 |

The imaginary part k is directly related to the absorption coefficient: `a = 4*pi*k/lambda`. For refined oils at visible wavelengths, k is small enough that absorption is negligible over meter-scale path lengths.

### Absorption in NIR

Petroleum oils show significant absorption bands in the near-infrared (NIR) due to C-H stretching overtones:
- Strong band near 1200 nm (third overtone of C-H stretch)
- Band near 1700 nm (second overtone)
- Strong absorption band near 2300 nm

At 1064 nm (Nd:YAG fundamental), `a_oil` is typically 0.5–5 m^-1 depending on composition and API gravity.

**Extension stub:** `OilMedium` accepts a user-supplied refractive-index dispersion table and computes absorption from the Kramers-Kronig relations if only the real part is provided.

---

## Brine Model

### Quan and Fry (1995) Extended Sellmeier

The refractive index of seawater and brine solutions is described by the Quan and Fry (1995) formula, valid for:
- Wavelength: 400–700 nm
- Temperature: 0–30°C
- Salinity: 0–35 g/kg (seawater)

The formula extends the standard Sellmeier dispersion to include salinity and temperature dependence:

```
n(lambda, T, S) = n_0 + n_1*S + n_2*T + n_3*T^2 + n_4/(lambda^2)
                + n_5*S/lambda^2 + n_6*T/lambda^2 + n_7*T^3
```

where lambda is in nanometres and T is in degrees Celsius. Coefficients:

```
n_0 = 1.31405
n_1 = 1.779e-4
n_2 = -1.05e-6
n_3 = 1.6e-8
n_4 = -2.02e-6
n_5 = 15.868
n_6 = 0.01155
n_7 = -0.00423
```

These are for pure NaCl solutions. The formula has been validated against measured data for seawater (mixed NaCl/MgCl2/Na2SO4); accuracy is approximately 1e-5 in n over the valid range.

**Reference:** Quan, X. and Fry, E.S. (1995). "Empirical equation for the index of refraction of seawater." *Applied Optics* 34(18), 3477–3480.

### Multi-salt extension

For brines containing KCl, CaCl2, or MgCl2 (common in oil-field produced water), the Quan-Fry formula underestimates n by up to 5e-4 at high concentrations. A more accurate model uses the partial molar refraction:

```
n_brine = n_water + sum_i [ (dn/dC_i) * C_i ]
```

where C_i is the concentration of salt i in g/L and `dn/dC_i` is the specific refractivity increment (approximately 1.7e-4 L/g for NaCl, 1.5e-4 for KCl, 2.3e-4 for CaCl2 at 532 nm).

```python
from photonator.materials import BrineMedium

# Oilfield brine: 150 g/L NaCl, 20 g/L CaCl2, 10 g/L KCl
medium = BrineMedium(
    nacl_g_per_l=150.0,
    cacl2_g_per_l=20.0,
    kcl_g_per_l=10.0,
    temperature_c=25.0,
    wavelength_nm=532.0
)
print(medium.refractive_index)   # approximately 1.378
```

---

## Mixture Model

For suspensions of particles in water (e.g., Maalox in water, oil droplets in seawater, phytoplankton in coastal water), the optical properties are combined by volume-fraction weighting.

**Volume-fraction weighting:**

```
a_mix = f_1 * a_1 + f_2 * a_2 + ... + f_k * a_k
b_mix = f_1 * b_1 + f_2 * b_2 + ... + f_k * b_k
```

where f_i is the volume fraction of component i (dimensionless, sum to 1) and a_i, b_i are its absorption and scattering coefficients (m^-1).

**Note:** Volume-fraction weighting is valid in the independent-scattering approximation (particles separated by much more than a wavelength). For dense suspensions (volume fraction > approximately 0.01), dependent scattering effects reduce b_mix below the linear prediction.

**Phase function of a mixture:**

The effective phase function of a mixture is the scattering-weighted average of the individual component phase functions:

```
p_mix(theta) = (b_1*f_1*p_1(theta) + b_2*f_2*p_2(theta) + ...) / b_mix
```

```python
from photonator.media import MixtureMedium
from photonator.materials import water_532nm, maalox_pf

water = water_532nm()            # HomogeneousMedium for pure water at 532 nm
maalox = maalox_medium(b=1.0)   # Maalox suspension, b = 1.0 m^-1

mixture = MixtureMedium([
    (water,  volume_fraction=0.99),
    (maalox, volume_fraction=0.01),
])
```

---

## Custom Material: Loading from File

Custom optical property tables can be loaded from HDF5 or CSV files with a standardized column layout.

**Required columns:**

| Column name | Units | Description |
|-------------|-------|-------------|
| `wavelength_nm` | nm | Wavelength grid |
| `mu_a_per_m` | m^-1 | Absorption coefficient a |
| `mu_s_per_m` | m^-1 | Scattering coefficient b |
| `g` | dimensionless | HG asymmetry parameter (used if no VSF column) |
| `n` | dimensionless | Real refractive index |

**Optional columns (for tabulated VSF, overrides g):**

| Column name | Units | Description |
|-------------|-------|-------------|
| `vsf_angle_rad` | rad | Scattering angle grid |
| `vsf_m_per_sr` | m^-1 sr^-1 | Volume scattering function values |

**CSV format (single wavelength or multiple wavelengths):**

```csv
wavelength_nm,mu_a_per_m,mu_s_per_m,g,n
532,0.0443,0.114,0.924,1.3370
550,0.0596,0.110,0.920,1.3362
```

**HDF5 format:**

```
/                           (root)
├── wavelength_nm           Dataset: float64, shape (N_wl,)
├── mu_a_per_m              Dataset: float64, shape (N_wl,)
├── mu_s_per_m              Dataset: float64, shape (N_wl,)
├── g                       Dataset: float64, shape (N_wl,)
├── n                       Dataset: float64, shape (N_wl,)
└── vsf/                    Group (optional, for tabulated VSF)
    ├── angle_rad           Dataset: float64, shape (N_angle,)
    └── vsf_m_per_sr        Dataset: float64, shape (N_wl, N_angle)
```

**Loading example:**

```python
from photonator.media import TabulatedMedium

# Load from HDF5
medium = TabulatedMedium.from_file(
    path='/data/my_ocean_water.h5',
    wavelength_nm=532.0     # interpolate to this wavelength
)

# Load from CSV
medium = TabulatedMedium.from_file(
    path='/data/my_suspension.csv',
    wavelength_nm=650.0
)
```

The `from_file` loader:
1. Reads the table and detects the file format from the extension.
2. Interpolates a, b, n to the requested wavelength.
3. If a `vsf/` group or VSF columns are present, builds a `UserDefinedPhaseFunction` from the tabulated VSF at the requested wavelength (with nearest-wavelength selection if exact wavelength is not in the table).
4. Otherwise, builds a `HenyeyGreensteinPF` using the interpolated g value.

---

## Extension Hooks: Spectral and Wideband Sources

For simulations with a spectrally broad source (e.g., pulsed white light, LED array, solar illumination), the `TabulatedMedium` class supports per-wavelength property tables without code changes.

**Wideband workflow:**

```python
from photonator.media import TabulatedMedium
from photonator.beams import BroadbandGaussianBeam   # emits photons at various wavelengths
from photonator import Simulation, Receiver

medium = TabulatedMedium.from_file('/data/coastal_water_400_700nm.h5')
beam = BroadbandGaussianBeam(
    waist_m=0.001,
    divergence_rad=7.5e-4,
    spectral_distribution='D65'   # CIE D65 daylight spectrum
)
rx = Receiver(position_xy=(0,0), aperture_m=0.1, fov_rad=np.pi/4)

sim = Simulation(
    beam=beam, medium=medium, receiver=rx,
    receiver_z_m=10.0, n_photons=1_000_000,
    spectral_bins_nm=np.arange(400, 701, 10)   # 10 nm bins
)
result = sim.run()

# result.spectral_power[i] = received power in wavelength bin i
```

Each photon packet carries a `wavelength_nm` attribute sampled from the source spectral distribution. At each step, `medium.absorption_coeff(wavelength_nm)`, `medium.scattering_coeff(wavelength_nm)`, and `medium.phase_function(wavelength_nm)` are called to retrieve wavelength-specific properties.

**AbstractMedium stub for wideband:**

```python
def emission_wavelength_nm(self, excitation_nm: float) -> float:
    """Sample an emission wavelength for inelastic processes.

    For fluorescent media, this returns a wavelength drawn from the
    fluorescence emission spectrum (a Stokes-shifted distribution).
    For Raman-active media, this returns the Raman-shifted wavelength.

    Default implementation raises NotImplementedError. Override in
    subclasses that support inelastic scattering.

    Parameters
    ----------
    excitation_nm : float
        Excitation (incident) photon wavelength in nanometres.

    Returns
    -------
    float
        Emission wavelength in nanometres.
    """
    raise NotImplementedError(
        "This medium does not support inelastic emission. "
        "Override emission_wavelength_nm() to add fluorescence or Raman."
    )
```
