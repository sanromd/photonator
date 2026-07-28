"""Wideband (spectral) simulation support — Phase 4.

The core engine is monochromatic per run; wideband sources are handled by
binning the source spectrum and running one monochromatic simulation per
bin, with the medium evaluated at each bin's wavelength via
``AbstractMedium.at_wavelength``.  This keeps every propagation loop
vectorized with scalar coefficients while producing a fully
wavelength-resolved received-power spectrum.

Example
-------
>>> spec = SpectralSimulation(
...     medium=Water(),
...     beam=GaussianBeam(w0_m=0.001),
...     phase_fn=HenyeyGreensteinPhaseFunction(g=0.93),
...     receiver=Receiver(receiver_z_m=8.0, aperture_m=1.0),
...     wavelengths_nm=[450, 500, 550, 600],
... )
>>> result = spec.run()
>>> result.power  # transmitted energy fraction per bin
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from photonator.beam.base import AbstractBeam
from photonator.core.receiver import Receiver
from photonator.media.base import AbstractMedium
from photonator.phase_functions.base import AbstractPhaseFunction
from photonator.simulation import Simulation, SimulationResult


@dataclass
class SpectralResult:
    """Wavelength-resolved output of a SpectralSimulation.

    Attributes
    ----------
    wavelengths_nm : bin centre wavelengths
    power : received energy fraction per bin — spectral weight ×
        (received power / launched photons) for that bin
    packets : detected photon count per bin
    total_power : sum of ``power`` over all bins
    bin_results : full per-bin SimulationResult objects
    """

    wavelengths_nm: NDArray[np.float64]
    power: NDArray[np.float64]
    packets: NDArray[np.int64]
    total_power: float = 0.0
    bin_results: list[SimulationResult] = field(default_factory=list)


class SpectralSimulation:
    """Run a wavelength-binned simulation of a wideband source.

    Parameters
    ----------
    medium : optical medium; evaluated per bin via ``at_wavelength``.
        Media without spectral data propagate identically in every bin.
    beam : beam profile initializer (spatial profile shared by all bins)
    phase_fn : phase function (shared by all bins)
    receiver : Receiver geometry; a fresh clone is used per bin
    wavelengths_nm : bin centre wavelengths (nm)
    spectral_weights : relative source energy per bin; normalised to sum
        to 1.  Defaults to a flat spectrum.
    n_photons : photons per bin per batch
    n_batches : batches per bin
    seed : master seed; each bin gets an independent stream
    """

    def __init__(
        self,
        medium: AbstractMedium,
        beam: AbstractBeam,
        phase_fn: AbstractPhaseFunction,
        receiver: Receiver,
        wavelengths_nm: NDArray[np.float64] | list[float],
        spectral_weights: NDArray[np.float64] | list[float] | None = None,
        n_photons: int = 100_000,
        n_batches: int = 1,
        seed: int = 0,
    ) -> None:
        self.medium = medium
        self.beam = beam
        self.phase_fn = phase_fn
        self.receiver = receiver
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        if self.wavelengths_nm.ndim != 1 or self.wavelengths_nm.size == 0:
            raise ValueError("wavelengths_nm must be a non-empty 1-D sequence")

        if spectral_weights is None:
            weights = np.ones_like(self.wavelengths_nm)
        else:
            weights = np.asarray(spectral_weights, dtype=np.float64)
            if weights.shape != self.wavelengths_nm.shape:
                raise ValueError("spectral_weights must match wavelengths_nm in length")
            if np.any(weights < 0):
                raise ValueError("spectral_weights must be non-negative")
        self.spectral_weights = weights / np.sum(weights)

        self.n_photons = n_photons
        self.n_batches = n_batches
        self.seed = seed

    def run(self) -> SpectralResult:
        """Run one monochromatic simulation per bin and aggregate."""
        n_bins = len(self.wavelengths_nm)
        power = np.zeros(n_bins, dtype=np.float64)
        packets = np.zeros(n_bins, dtype=np.int64)
        bin_results: list[SimulationResult] = []

        for i, wl in enumerate(self.wavelengths_nm):
            medium_wl = self.medium.at_wavelength(float(wl))
            sim = Simulation(
                medium=medium_wl,
                beam=self.beam,
                phase_fn=self.phase_fn,
                receiver=self.receiver.clone(),
                n_photons=self.n_photons,
                n_batches=self.n_batches,
                seed=self.seed + 7919 * i,   # independent stream per bin
            )
            res = sim.run()
            bin_results.append(res)
            # Energy fraction reaching the receiver in this bin
            power[i] = self.spectral_weights[i] * res.total_power / max(res.n_photons, 1)
            packets[i] = res.total_packets

        return SpectralResult(
            wavelengths_nm=self.wavelengths_nm,
            power=power,
            packets=packets,
            total_power=float(np.sum(power)),
            bin_results=bin_results,
        )
