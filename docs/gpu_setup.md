# GPU Setup Guide

This guide covers configuring photonator to use NVIDIA GPU acceleration via the Numba CUDA and CuPy backends. GPU execution can reduce simulation time by 10–100x for large photon counts compared to single-core CPU.

---

## Requirements

### Hardware

- NVIDIA GPU with compute capability 5.0 or later (Maxwell architecture, 2014+)
- Recommended: compute capability 7.0+ (Volta: V100, Ampere: A100/RTX 3090, Ada: RTX 4090)
- Minimum 4 GB GPU VRAM for typical photon batch sizes (1M photons × 8 floats × 8 bytes = 64 MB; simulation requires additional working memory for random states and intermediate arrays)

### CUDA Toolkit

- CUDA 11.x or CUDA 12.x (both are supported)
- Install from the NVIDIA developer website: https://developer.nvidia.com/cuda-downloads
- Verify installation:

```bash
nvcc --version
nvidia-smi
```

The `nvidia-smi` command should report the driver version and GPU name. `nvcc --version` should report the toolkit version.

### Python

- Python 3.10 or later
- 64-bit CPython (not PyPy; Numba CUDA requires CPython)

---

## Installing Numba

Numba provides the JIT-compiled CPU backend and the CUDA kernel backend for photonator.

```bash
pip install numba
```

For conda environments:

```bash
conda install numba cudatoolkit -c conda-forge
```

**CUDA toolkit path (if nvcc is not on PATH):**

Numba needs to find the CUDA toolkit libraries. If installation succeeds but Numba cannot find CUDA, set:

```bash
export CUDA_HOME=/usr/local/cuda         # typical Linux path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

On Windows:

```
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
```

Add these to your shell profile (`.bashrc`, `.zshrc`, or Windows environment variables) for persistence.

**Verify Numba CUDA:**

```bash
python -c "from numba import cuda; print(cuda.gpus)"
```

Expected output (one or more GPUs detected):

```
<Managed Device 0 on Platform 0>
```

If you see `CudaSupportError` or an empty list, check that the CUDA toolkit is installed and the paths above are set correctly.

---

## Installing CuPy

CuPy provides GPU-accelerated NumPy-compatible array operations. The package suffix must match your installed CUDA version:

| CUDA version | pip package |
|--------------|-------------|
| CUDA 11.2–11.8 | `cupy-cuda11x` |
| CUDA 12.x | `cupy-cuda12x` |
| Any (slower build) | `cupy` |

```bash
# For CUDA 12.x (most common with modern hardware):
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x
```

**Verify CuPy:**

```bash
python -c "import cupy as cp; a = cp.array([1,2,3]); print(a)"
```

Expected output: `[1 2 3]` (printed from GPU memory).

If CuPy cannot find CUDA libraries, check `CUDA_PATH` / `CUDA_HOME` as described above.

---

## Device Check

Before running a simulation, verify that Python can see the GPU:

```bash
# Numba device list:
python -c "from numba import cuda; print(cuda.gpus)"

# CuPy device info:
python -c "import cupy; print(cupy.cuda.Device(0).attributes)"

# photonator built-in check:
python -m photonator.check_gpu
```

The photonator GPU check script reports:

```
photonator GPU capability check
================================
CUDA available (Numba):   YES
  Device 0: NVIDIA GeForce RTX 4090 (compute 8.9)
  Free VRAM: 23.5 / 24.0 GB

CuPy available:           YES
  CuPy version: 13.2.0
  CUDA version: 12.4

Recommended backend:      cupy
```

---

## Backend Selection in Simulation

Pass the `backend` argument to `Simulation`:

```python
from photonator import Simulation, Receiver
from photonator.beams import GaussianBeam
from photonator.media import HomogeneousMedium
from photonator.phase_functions import PetzoldPhaseFunction

beam   = GaussianBeam(waist_m=0.001, divergence_rad=7.5e-4)
medium = HomogeneousMedium(a_per_m=0.025, b_per_m=0.125,
                           phase_function=PetzoldPhaseFunction.from_preset('petzold_clear'))
rx     = Receiver(position_xy=(0,0), aperture_m=0.8, fov_rad=1.5708)

# --- CPU (NumPy) ---
sim_cpu = Simulation(beam=beam, medium=medium, receiver=rx,
                     receiver_z_m=106.67, n_photons=1_000_000,
                     backend='numpy', seed=42)

# --- GPU (CuPy) ---
sim_gpu = Simulation(beam=beam, medium=medium, receiver=rx,
                     receiver_z_m=106.67, n_photons=1_000_000,
                     backend='cupy', seed=42)

# --- GPU (Numba CUDA) ---
sim_numba = Simulation(beam=beam, medium=medium, receiver=rx,
                       receiver_z_m=106.67, n_photons=1_000_000,
                       backend='numba', device='cuda', seed=42)

result = sim_gpu.run()
```

### Backend-specific notes

**`backend='numpy'`:** Single-threaded CPU. Uses standard NumPy arrays. Works on any machine without NVIDIA hardware. Default when no backend is specified.

**`backend='numba'`:** Uses Numba JIT compilation. On CPU, this gives 2–5x speedup over NumPy for the propagation loop. On CUDA (`device='cuda'`), the propagation loop is compiled to a CUDA kernel; each GPU thread processes one photon. The CDF lookup uses a texure-memory-backed array for fast random access.

**`backend='cupy'`:** Uses CuPy GPU arrays throughout. The entire photon state array lives in GPU VRAM. Array operations (path sampling, position update, direction rotation) are dispatched to cuBLAS and cuRAND. Transfer back to host memory occurs only when returning the `SimulationResult`.

**Multi-GPU:** To use a specific GPU, set the CUDA device before creating the Simulation:

```python
import cupy as cp
cp.cuda.Device(1).use()   # Use GPU index 1
sim = Simulation(..., backend='cupy')
```

---

## CPU Fallback Behavior

When a GPU backend is requested but no compatible GPU is found, photonator automatically falls back to the NumPy CPU backend:

```python
sim = Simulation(..., backend='cupy')
# If no GPU: logs a warning and uses numpy
```

Log output:

```
WARNING photonator.backends: CuPy requested but no CUDA device found.
         Falling back to numpy backend. Install cupy-cuda12x and ensure
         CUDA drivers are installed.
```

To suppress the fallback and raise an error instead (useful for CI/CD pipelines that must use GPU):

```python
sim = Simulation(..., backend='cupy', require_gpu=True)
# Raises: photonator.BackendError: GPU backend 'cupy' not available.
```

---

## Performance Expectations

### Reference configuration

- 1 million photons, 16 optical attenuation lengths, clear ocean (c = 0.15 m^-1)
- Petzold clear VSF phase function
- Single receiver, wide FOV

| Hardware | Backend | Approximate wall time |
|----------|---------|----------------------|
| AMD Ryzen 9 5950X (16 cores) | numpy (1 core) | 8–12 minutes |
| AMD Ryzen 9 5950X | numba (1 core JIT) | 3–5 minutes |
| NVIDIA RTX 3080 (10 GB) | cupy | 25–45 seconds |
| NVIDIA A100 (40 GB) | cupy | 10–20 seconds |
| NVIDIA RTX 4090 (24 GB) | cupy | 8–15 seconds |

GPU speedup over single-core CPU: 10–80x depending on hardware generation.

**Guideline:** 1 million photons should complete in under 60 seconds on any CUDA-capable GPU from 2018 or later (Turing architecture, RTX 20-series or better). On CPU (single core), expect under 10 minutes for the same 1 million photons.

### Scaling with photon count

Execution time scales approximately linearly with n_photons for a fixed receiver depth, since the number of propagation steps per photon scales with the number of attenuation lengths traversed (which is fixed when receiver_z_m is fixed). Memory usage scales linearly as 8 × 8 bytes × n_photons = 64 bytes per photon. For 10 million photons, the photon state array requires 640 MB of VRAM.

### GPU memory limit

If n_photons exceeds available VRAM, photonator automatically splits into batches:

```python
sim = Simulation(..., n_photons=10_000_000, backend='cupy')
# If VRAM < 640 MB: automatically runs as 10 batches of 1M each
```

The split is transparent to the user; results from all batches are accumulated before returning.

---

## Troubleshooting

**`ImportError: No module named 'cupy'`**

Install the correct cupy package for your CUDA version. Check CUDA version with `nvcc --version` or `nvidia-smi`.

**`CudaDriverError: Call to cuInit results in CUDA_ERROR_NO_DEVICE`**

No NVIDIA GPU detected. Check `nvidia-smi`. If on a headless server, ensure the NVIDIA driver is loaded: `sudo modprobe nvidia`.

**`numba.cuda.cudadrv.error.CudaDriverError: CUDA initialized before forking`**

Do not create CUDA contexts in the parent process before using `multiprocessing`. Use `backend='numpy'` for multi-process parallelism, or run one Simulation per GPU process.

**Slow first run**

Numba JIT compilation occurs on the first call; subsequent calls are fast. CuPy also compiles kernels on first use. Add a warmup call with a small `n_photons=1000` before benchmarking:

```python
sim_warmup = Simulation(..., n_photons=1000, backend='cupy')
sim_warmup.run()   # triggers compilation

sim_bench = Simulation(..., n_photons=1_000_000, backend='cupy')
import time
t0 = time.perf_counter()
result = sim_bench.run()
print(f"Elapsed: {time.perf_counter() - t0:.1f} s")
```
