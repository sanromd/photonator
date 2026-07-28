"""Monte Carlo photon transport simulator for scattering/absorbing liquids."""

from photonator.simulation import Simulation, SimulationResult
from photonator.spectral import SpectralResult, SpectralSimulation

__version__ = "2.1.0"
__all__ = ["Simulation", "SimulationResult", "SpectralSimulation", "SpectralResult"]
