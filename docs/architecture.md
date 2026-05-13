# Architecture Reference

## High-Level Design Goals

The Python refactor of photonator is guided by four principles:

1. **Vectorized NumPy core:** All per-photon operations are expressed as array operations over the entire active-photon batch simultaneously. There are no Python-level for-loops over individual photons. This enables the JIT compiler (Numba) and GPU array library (CuPy) to operate on dense numerical arrays with minimal Python overhead.

2. **Pluggable components:** Beam sources, phase functions, and media are defined by abstract base classes. Switching from a Gaussian beam to a Bessel beam, or from Petzold VSF to Henyey-Greenstein, requires only changing the object passed to the Simulation constructor — no modifications to the core propagation loop.

3. **GPU-ready from the start:** The propagation loop is written against a NumPy-compatible array API. Swapping `import numpy as np` for `import cupy as cp` (or using Numba CUDA kernels) requires no algorithmic changes. The backend is selected at runtime by a single `backend` argument to Simulation.

4. **Testable and parameterized:** All physical parameters are arguments to class constructors, not global variables. Every component can be instantiated and tested in isolation. The Simulation.run() method is deterministic given a fixed RNG seed.

---

## Class Hierarchy

```
photonator/
├── beams/
│   ├── AbstractBeam              (ABC)
│   ├── GaussianBeam              (paraxial Gaussian, thin-lens divergence)
│   ├── FlatTopBeam               (uniform disk)
│   ├── BesselBeam                (stub)
│   └── LaguerreGaussianBeam      (stub)
│
├── phase_functions/
│   ├── AbstractPhaseFunction     (ABC)
│   ├── HenyeyGreensteinPF        (analytic CDF, closed-form sampling)
│   ├── PetzoldPhaseFunction      (tabulated VSF, cumtrapz CDF)
│   └── UserDefinedPhaseFunction  (from_file: CSV or HDF5)
│
├── media/
│   ├── AbstractMedium            (ABC)
│   ├── HomogeneousMedium         (constant a, b, c, phase_function)
│   └── TabulatedMedium           (wavelength-indexed property tables)
│
├── core/
│   ├── PhotonBatch               (N×8 array wrapper with named columns)
│   ├── Receiver                  (Fresnel + aperture + FOV + Welford stats)
│   └── Simulation                (orchestrator; run() → batched propagation)
│
└── backends/
    ├── numpy_backend             (default, CPU)
    ├── numba_backend             (Numba JIT, CPU or CUDA)
    └── cupy_backend              (CuPy, NVIDIA GPU)
```

### AbstractBeam

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class AbstractBeam(ABC):
    """Base class for photon source beam profiles.

    All subclasses must implement initialize(), which returns the initial
    state of a batch of photon packets.
    """

    @abstractmethod
    def initialize(
        self, n_photons: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Generate initial photon positions and direction cosines.

        Parameters
        ----------
        n_photons : int
            Number of photon packets to initialize.
        rng : np.random.Generator
            Random number generator (passed in for reproducibility).

        Returns
        -------
        x, y, z : np.ndarray, shape (n_photons,)
            Initial positions (m).
        ux, uy, uz : np.ndarray, shape (n_photons,)
            Initial direction cosines (unit vectors).
        """
```

### AbstractPhaseFunction

```python
class AbstractPhaseFunction(ABC):
    """Base class for scattering phase functions.

    Subclasses implement angle sampling and (optionally) inelastic
    scattering. The default inelastic_yield returns 0.0, making all
    interactions elastic.
    """

    @abstractmethod
    def sample_angle(
        self, n: int, rng: np.random.Generator, wavelength_nm: float = 532.0
    ) -> np.ndarray:
        """Sample polar scattering angles from the phase function CDF.

        Parameters
        ----------
        n : int
            Number of angles to sample.
        rng : np.random.Generator
            Random number generator.
        wavelength_nm : float
            Photon wavelength in nanometres (for spectral variants).

        Returns
        -------
        theta : np.ndarray, shape (n,)
            Polar scattering angles in radians, in [0, pi].
        """

    def inelastic_yield(self, wavelength_nm: float = 532.0) -> float:
        """Probability of inelastic scattering per interaction.

        Override in subclasses for fluorescent or Raman-active media.
        Default implementation returns 0.0 (elastic scattering only).
        """
        return 0.0

    def emission_wavelength_nm(self, excitation_nm: float) -> float:
        """Sample the emission wavelength for an inelastic event.

        Override this method together with inelastic_yield() to implement
        fluorescence or Raman scattering. Raises NotImplementedError by
        default.
        """
        raise NotImplementedError(
            "Inelastic emission is not implemented for this phase function."
        )
```

### AbstractMedium

```python
class AbstractMedium(ABC):
    """Base class for optical media.

    Encapsulates the wavelength-dependent optical properties: absorption
    coefficient a (m^-1), scattering coefficient b (m^-1), and the phase
    function. The total attenuation coefficient is c = a + b.
    """

    @abstractmethod
    def absorption_coeff(self, wavelength_nm: float = 532.0) -> float:
        """Return absorption coefficient a (1/m)."""

    @abstractmethod
    def scattering_coeff(self, wavelength_nm: float = 532.0) -> float:
        """Return scattering coefficient b (1/m)."""

    def attenuation_coeff(self, wavelength_nm: float = 532.0) -> float:
        """Return total attenuation coefficient c = a + b (1/m)."""
        return self.absorption_coeff(wavelength_nm) + self.scattering_coeff(wavelength_nm)

    def albedo(self, wavelength_nm: float = 532.0) -> float:
        """Return single-scattering albedo omega_0 = b/c."""
        b = self.scattering_coeff(wavelength_nm)
        c = self.attenuation_coeff(wavelength_nm)
        return b / c

    @abstractmethod
    def phase_function(self, wavelength_nm: float = 532.0) -> AbstractPhaseFunction:
        """Return the phase function for this medium at the given wavelength."""

    def inelastic_yield(self, wavelength_nm: float = 532.0) -> float:
        """Probability of inelastic scattering per interaction. Stub; returns 0."""
        return 0.0

    def emission_wavelength_nm(self, excitation_nm: float) -> float:
        """Sample emission wavelength for inelastic events. Stub; raises."""
        raise NotImplementedError
```

### PhotonBatch

```python
class PhotonBatch:
    """Wrapper around the N×8 photon state array.

    Column layout:
        0: x      - x position (m)
        1: y      - y position (m)
        2: z      - z position (m)
        3: ux     - x direction cosine
        4: uy     - y direction cosine
        5: uz     - z direction cosine
        6: weight - photon packet weight [0, 1]
        7: status - 1=active, 0=received, -1=terminated

    All operations act on the active subset (status == 1) to avoid
    wasted computation on terminated photons.
    """

    COLS = dict(x=0, y=1, z=2, ux=3, uy=4, uz=5, weight=6, status=7)
    STATUS_ACTIVE    = 1
    STATUS_RECEIVED  = 0
    STATUS_TERMINATED = -1
```

### Receiver

```python
class Receiver:
    """Planar receiver with Fresnel interface, aperture, FOV, and statistics.

    Parameters
    ----------
    position_xy : tuple[float, float]
        Center of receiver in the z=receiver_z plane (m).
    aperture_m : float
        Diameter of the receiver aperture (m).
    fov_rad : float
        Full field-of-view angle (rad). Photons outside half-angle are rejected.
    n_water : float
        Refractive index of the incident medium (default 1.33).
    n_air : float
        Refractive index of the exit medium (default 1.0).
    fov_weighting : str
        Angular sensitivity model: 'uniform', 'gaussian', or 'two_term_gaussian'.
    """
```

### Simulation

```python
class Simulation:
    """Top-level simulation orchestrator.

    Parameters
    ----------
    beam : AbstractBeam
        Photon source.
    medium : AbstractMedium
        Propagation medium.
    receiver : Receiver
        Detector configuration.
    receiver_z_m : float
        Axial position of the receiver plane (m).
    n_photons : int
        Number of photon packets per batch.
    n_batches : int
        Number of independent batches (results are accumulated).
    roulette_threshold : float
        Weight below which roulette is applied.
    roulette_multiplier : int
        Inverse of survival probability for roulette (default 10).
    seed : int or None
        RNG seed for reproducibility.
    backend : str
        Compute backend: 'numpy' | 'numba' | 'cupy'.
    wavelength_nm : float
        Simulation wavelength in nanometres (default 532.0).
    """

    def run(self) -> SimulationResult:
        """Execute all batches and return aggregated results."""
```

---

## Data Flow

```
Simulation.run()
    │
    ├── rng = np.random.default_rng(seed)
    │
    ├── for batch in range(n_batches):
    │       │
    │       ├── PhotonBatch = beam.initialize(n_photons, rng)
    │       │   Returns (x, y, z, ux, uy, uz) arrays; weight=1, status=active
    │       │
    │       ├── cdf, angles = medium.phase_function(wavelength_nm).precompute()
    │       │
    │       ├── Propagation loop (while any photon active):
    │       │   │
    │       │   ├── Select active mask: idx = (status == 1)
    │       │   │
    │       │   ├── r = -log(U) / c     (vectorized exponential sampling)
    │       │   │
    │       │   ├── Advance positions: x[idx] += r*ux[idx], etc.
    │       │   │
    │       │   ├── Detect receiver crossing: z_new >= receiver_z
    │       │   │   ├── Ray-trace to plane, record (x, y, ux, uy, uz, weight)
    │       │   │   └── status[hit] = RECEIVED
    │       │   │
    │       │   ├── Apply implicit absorption: weight[idx] *= omega_0
    │       │   │
    │       │   ├── Roulette: weight[idx] < threshold?
    │       │   │   ├── Terminate (prob = 1 - 1/M): status = TERMINATED
    │       │   │   └── Boost (prob = 1/M): weight *= M
    │       │   │
    │       │   ├── theta = np.interp(rng.random(n_active), cdf, angles)
    │       │   ├── phi   = 2*pi * rng.random(n_active)
    │       │   │
    │       │   └── Update direction cosines (vectorized rotation)
    │       │
    │       └── receiver.detect(photons_at_plane)
    │           ├── Critical angle test
    │           ├── Fresnel transmission
    │           ├── Aperture test
    │           ├── FOV test
    │           ├── Angular weighting
    │           └── Welford accumulation
    │
    └── Return SimulationResult(power, ph_cnt, angle_stats, dist_stats, ...)
```

---

## ASCII Class Diagram

```
                    ┌─────────────────────┐
                    │    Simulation       │
                    │─────────────────────│
                    │ beam: AbstractBeam  │
                    │ medium: AbstractMed │
                    │ receiver: Receiver  │
                    │ backend: str        │
                    │─────────────────────│
                    │ run() → Result      │
                    └─────────┬───────────┘
                              │ uses
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           v                  v                  v
  ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
  │  AbstractBeam   │ │AbstractMedium│ │   Receiver    │
  │  (ABC)          │ │  (ABC)       │ │───────────────│
  │─────────────────│ │──────────────│ │ aperture_m    │
  │ initialize()    │ │ absorption() │ │ fov_rad       │
  └────────┬────────┘ │ scattering() │ │ n_water       │
           │          │ albedo()     │ │───────────────│
    ┌──────┴─────┐    │ phase_func() │ │ detect()      │
    │            │    └──────┬───────┘ └───────────────┘
    v            v           │
GaussianBeam  FlatTopBeam    │ returns
(+Bessel, LG  (stubs)        v
 stubs)            ┌──────────────────────┐
                   │AbstractPhaseFunction │
                   │  (ABC)               │
                   │──────────────────────│
                   │ sample_angle()       │
                   │ inelastic_yield()    │
                   │ emission_wl_nm()     │
                   └──────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              v               v               v
   HenyeyGreenstein    PetzoldPF       UserDefinedPF
   PF (analytic CDF)   (tabulated)    (from_file)

                    ┌─────────────────────┐
                    │   PhotonBatch       │
                    │─────────────────────│
                    │ data: ndarray N×8   │
                    │ active_mask()       │
                    │ x, y, z properties  │
                    │ ux, uy, uz props    │
                    │ weight property     │
                    │ status property     │
                    └─────────────────────┘
```

---

## Extension Points

### Adding a new beam profile

1. Create `photonator/beams/my_beam.py`.
2. Subclass `AbstractBeam` and implement `initialize(n_photons, rng)`.
3. Return six arrays: x, y, z, ux, uy, uz (each shape `(n_photons,)`).
4. Register in `photonator/beams/__init__.py`.

Example skeleton:

```python
from photonator.beams.abstract import AbstractBeam
import numpy as np

class FlatTopBeam(AbstractBeam):
    """Uniform circular disk illumination with zero divergence."""

    def __init__(self, radius_m: float) -> None:
        self.radius_m = radius_m

    def initialize(self, n_photons: int, rng: np.random.Generator):
        # Sample uniform disk by rejection or by sqrt(U)*R, phi=2*pi*V
        r   = self.radius_m * np.sqrt(rng.random(n_photons))
        phi = 2 * np.pi * rng.random(n_photons)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z  = np.zeros(n_photons)
        ux = np.zeros(n_photons)
        uy = np.zeros(n_photons)
        uz = np.ones(n_photons)
        return x, y, z, ux, uy, uz
```

### Adding a new medium

1. Create `photonator/media/my_medium.py`.
2. Subclass `AbstractMedium` and implement `absorption_coeff`, `scattering_coeff`, and `phase_function`.
3. Optionally override `inelastic_yield` and `emission_wavelength_nm` for fluorescent or Raman media.

```python
from photonator.media.abstract import AbstractMedium
from photonator.phase_functions import HenyeyGreensteinPF

class HarborWater(AbstractMedium):
    def absorption_coeff(self, wavelength_nm=532.0): return 0.366
    def scattering_coeff(self, wavelength_nm=532.0): return 1.824
    def phase_function(self, wavelength_nm=532.0):
        return HenyeyGreensteinPF(g=0.924)
```

### Adding a new phase function

1. Create `photonator/phase_functions/my_pf.py`.
2. Subclass `AbstractPhaseFunction` and implement `sample_angle(n, rng, wavelength_nm)`.
3. The method must return an array of shape `(n,)` with values in `[0, pi]`.

For a tabulated VSF:

```python
import numpy as np
from photonator.phase_functions.abstract import AbstractPhaseFunction

class MyVSFPhaseFunction(AbstractPhaseFunction):
    def __init__(self, angle_rad: np.ndarray, vsf: np.ndarray) -> None:
        cdf = np.cumulative_trapezoid(vsf * np.sin(angle_rad), angle_rad)
        cdf /= cdf[-1]
        self._cdf = cdf
        self._angles = angle_rad[1:]   # cumulative_trapezoid drops first point

    def sample_angle(self, n, rng, wavelength_nm=532.0):
        u = rng.random(n)
        return np.interp(u, self._cdf, self._angles)
```

---

## Backend Switching

The backend determines which array library executes the numerical computation.

### numpy (default, CPU)

```python
sim = Simulation(beam=beam, medium=medium, receiver=rx,
                 receiver_z_m=10.0, n_photons=100_000,
                 backend='numpy')
```

No additional dependencies beyond NumPy and SciPy.

### numba (JIT-compiled CPU or CUDA)

```python
sim = Simulation(..., backend='numba')
```

Requires `pip install numba`. On CPU, Numba JIT-compiles the inner loop to native machine code with SIMD and removes Python overhead. With a CUDA-capable GPU, add `device='cuda'`:

```python
sim = Simulation(..., backend='numba', device='cuda')
```

The propagation kernel is compiled as a `@cuda.jit` function. Each GPU thread handles one photon packet. Thread blocks are sized to maximize occupancy for the target GPU.

### cupy (CuPy GPU arrays)

```python
sim = Simulation(..., backend='cupy')
```

Requires `pip install cupy-cuda12x` (adjust suffix for your CUDA version). The entire photon state array lives in GPU memory as a `cupy.ndarray`. Array operations are dispatched to cuBLAS and custom CUDA kernels. Data transfer to host occurs only when returning `SimulationResult`.

### CPU fallback

When `backend='numba'` or `backend='cupy'` is requested but no GPU is available, the Simulation class automatically falls back to the NumPy CPU backend and logs a warning:

```
WARNING: GPU backend 'cupy' requested but no CUDA device detected. Falling back to 'numpy'.
```

This ensures that code written for GPU runs correctly on CPU-only machines (at reduced speed) without code changes.

---

## Inelastic and Wideband Extension Stubs

### In AbstractPhaseFunction

The `inelastic_yield` and `emission_wavelength_nm` methods are defined in `AbstractPhaseFunction` with no-op defaults. To implement Raman scattering in seawater:

```python
class RamanSeawaterPF(AbstractPhaseFunction):
    RAMAN_STOKES_SHIFT_NM = 3400   # cm^-1 Raman shift for O-H stretch

    def sample_angle(self, n, rng, wavelength_nm=532.0):
        # Raman is nearly isotropic for water (depolarization ratio ~0.17)
        # Use a slightly anisotropic phase function
        cos_theta = 1 - 2 * rng.random(n)
        return np.arccos(cos_theta)

    def inelastic_yield(self, wavelength_nm=532.0):
        # Raman cross-section for water: approximately 2e-4 relative to scattering
        return 2e-4

    def emission_wavelength_nm(self, excitation_nm):
        # Stokes Raman shift: 1/lambda_emission = 1/lambda_excitation - shift_wavenumber/1e7
        shift_per_cm = 3400
        inv_ex = 1.0 / excitation_nm  # nm^-1
        inv_em = inv_ex - shift_per_cm / 1e7
        return 1.0 / inv_em
```

### In AbstractMedium

The `inelastic_yield` and `emission_wavelength_nm` stubs in `AbstractMedium` delegate to the phase function by default, but can be overridden at the medium level (e.g., to add bulk fluorescence independent of scattering):

```python
class FluorescentMedium(HomogeneousMedium):
    def __init__(self, *args, fluorophore_conc_m: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.fluorophore_conc_m = fluorophore_conc_m

    def inelastic_yield(self, wavelength_nm=532.0):
        # Absorption cross-section of fluorophore times concentration
        sigma = 1e-20  # m^2, placeholder
        return sigma * self.fluorophore_conc_m

    def emission_wavelength_nm(self, excitation_nm):
        # Gaussian fluorescence spectrum centered at 560 nm
        mu = 560.0
        sigma_nm = 20.0
        return np.random.normal(mu, sigma_nm)
```
