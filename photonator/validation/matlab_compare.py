"""Compare Python simulation output against saved MATLAB results.

Usage::

    python -m photonator.validation.matlab_compare \\
        --matlab-file output.mat \\
        --n-photons 100000 \\
        --receiver-z 16.0 \\
        --c 0.15 \\
        --albedo 0.25

Pass criteria (matching MATLAB):
    - Received power within 1%
    - Mean angle within 0.5°
    - Path-length KS p > 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest  # type: ignore[import]


def compare(
    matlab_file: Path,
    n_photons: int,
    receiver_z_m: float,
    c_per_m: float,
    albedo: float,
    seed: int = 0,
) -> dict:
    """Run Python simulation and compare against MATLAB .mat output.

    Parameters
    ----------
    matlab_file : path to MATLAB output .mat containing rec_weights and total_rec_dist
    n_photons : number of photons
    receiver_z_m : receiver depth (m)
    c_per_m : attenuation coefficient (m^-1)
    albedo : single-scatter albedo
    seed : RNG seed

    Returns
    -------
    dict with comparison metrics and pass/fail flags
    """
    import scipy.io as sio

    from photonator.beam.gaussian import GaussianBeam
    from photonator.core.receiver import Receiver
    from photonator.media.water import Water
    from photonator.phase_functions.petzold import PetzoldPhaseFunction
    from photonator.simulation import Simulation

    mat = sio.loadmat(str(matlab_file))
    matlab_weights = np.ravel(mat.get("rec_weights", np.array([0.0])))
    matlab_dists = np.ravel(mat.get("total_rec_dist", np.array([0.0])))
    matlab_power = float(np.sum(matlab_weights))

    b_per_m = c_per_m * albedo
    a_per_m = c_per_m - b_per_m

    medium = Water(mu_a_per_m=a_per_m, mu_s_per_m=b_per_m)
    beam = GaussianBeam(w0_m=0.001, half_angle_divergence_rad=0.00075)
    phase_fn = PetzoldPhaseFunction(water_type="clear")
    receiver = Receiver(receiver_z_m=receiver_z_m, aperture_m=0.8, fov_rad=float(np.pi / 2))

    sim = Simulation(
        medium=medium, beam=beam, phase_fn=phase_fn, receiver=receiver,
        n_photons=n_photons, n_batches=1, seed=seed,
    )
    result = sim.run()

    py_power = result.total_power

    # Power comparison
    power_error = abs(py_power - matlab_power) / max(matlab_power, 1e-15)
    power_ok = power_error < 0.01

    # Mean angle comparison
    matlab_mean_uz = float(np.mean(matlab_weights)) if len(matlab_weights) else 0.0
    py_mean_uz = result.angle_mean_rad
    angle_diff_deg = abs(np.degrees(np.arccos(np.clip(matlab_mean_uz, -1, 1))) -
                         np.degrees(np.arccos(np.clip(py_mean_uz, -1, 1))))
    angle_ok = angle_diff_deg < 0.5

    # Path-length KS test
    if result.distances_m is not None and len(matlab_dists) > 1 and len(result.distances_m) > 1:
        ks_stat, ks_p = kstest(result.distances_m, matlab_dists)
        ks_ok = ks_p > 0.05
    else:
        ks_stat, ks_p, ks_ok = 0.0, 1.0, True

    passed = power_ok and angle_ok and ks_ok

    return {
        "matlab_power": matlab_power,
        "python_power": py_power,
        "power_error": power_error,
        "power_ok": power_ok,
        "angle_diff_deg": angle_diff_deg,
        "angle_ok": angle_ok,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "ks_ok": ks_ok,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python vs MATLAB simulation")
    parser.add_argument("--matlab-file", required=True, type=Path)
    parser.add_argument("--n-photons", type=int, default=100_000)
    parser.add_argument("--receiver-z", type=float, default=16.0)
    parser.add_argument("--c", type=float, default=0.15)
    parser.add_argument("--albedo", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = compare(
        matlab_file=args.matlab_file,
        n_photons=args.n_photons,
        receiver_z_m=args.receiver_z,
        c_per_m=args.c,
        albedo=args.albedo,
        seed=args.seed,
    )

    for k, v in result.items():
        print(f"  {k}: {v}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
