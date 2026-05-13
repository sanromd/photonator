"""Abstract base class for beam profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from photonator.core.photon import PhotonBatch


class AbstractBeam(ABC):
    """Base class for all beam profile initializers.

    Subclasses implement :meth:`initialize` to populate a :class:`PhotonBatch`
    with initial positions (x_m, y_m, z_m=0) and direction cosines (ux, uy, uz).
    """

    @abstractmethod
    def initialize(self, n_photons: int, rng: np.random.Generator) -> PhotonBatch:
        """Create a PhotonBatch with initial positions and directions.

        Parameters
        ----------
        n_photons : number of photons to initialise
        rng : NumPy random generator

        Returns
        -------
        PhotonBatch with all photons active, weight=1, z=0
        """
