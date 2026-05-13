"""Project-level physics constants.

All values sourced from scipy.constants where available.
"""

import scipy.constants as _sc

# Speed of light in vacuum (m/s)
c_vacuum_m_per_s: float = _sc.speed_of_light

# Refractive indices (dimensionless)
N_AIR: float = 1.0
N_WATER: float = 1.33          # seawater at visible wavelengths
N_WINDOW_POLYCARBONATE: float = 1.585

# Derived: critical angle cosine for water→air interface
# sin(theta_c) = n_air / n_water  →  cos(theta_c) = sqrt(1 - (n_air/n_water)^2)
import math as _math
CRIT_ANG_COS_WATER_AIR: float = _math.sqrt(1.0 - (N_AIR / N_WATER) ** 2)

# Roulette constants (matching original MATLAB implementation)
ROULETTE_CONST: int = 10          # photon survives with probability 1/ROULETTE_CONST
ROULETTE_CONST_INV: float = 1.0 / ROULETTE_CONST

# Minimum weight thresholds (MATLAB-faithful, keyed by prob_of_survival)
# prob_of_survival = b/c = (c-a)/c
def min_weight_for_albedo(albedo: float) -> float:
    """Return the roulette minimum-weight threshold for a given single-scatter albedo."""
    if albedo >= 0.90:
        return 1e-4
    if albedo >= 0.8299:
        return 1e-5
    if albedo >= 0.70:
        return 1e-5
    return 1e-7
