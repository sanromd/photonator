"""Fluorescent medium: any host medium doped with a bulk fluorophore.

The fluorophore absorbs at the excitation wavelength and re-emits at
``emission_wavelength_nm`` with quantum yield ``quantum_yield``.

The propagation loop reads ``medium.inelastic_yield`` and
``medium.emission_wavelength_nm`` to redirect emitted photons once
Phase 3 (inelastic propagation) is active.  Until then, the fluorophore
contribution simply adds to ``mu_a_per_m`` — absorbed photons are lost.
"""

from __future__ import annotations

from photonator.media.base import AbstractMedium


class FluorescentMedium(AbstractMedium):
    """Host medium doped with a homogeneous fluorophore.

    Parameters
    ----------
    host : AbstractMedium
        Background (undoped) medium.  All scattering and non-fluorophore
        absorption comes from here.
    mu_a_ex_per_m : float
        Fluorophore absorption coefficient at the excitation wavelength
        (m⁻¹).  This is added to ``host.mu_a_per_m``.
    emission_wavelength_nm : float
        Peak emission wavelength (nm).
    quantum_yield : float
        Fraction of absorbed fluorophore photons re-emitted (0–1).
        Stored as ``inelastic_yield`` and used by the propagation loop
        to weight fluorescent output.
    stokes_shift_nm : float
        Stokes shift (nm).  Purely informational — ``emission_wavelength_nm``
        should already include the shift.
    wavelength_nm : float | None
        Excitation wavelength; forwarded from host if None.

    Notes
    -----
    Classmethod ``from_concentration`` builds the medium from molar
    concentration + molar extinction coefficient, which is more natural
    for fluorescent dye solutions.
    """

    def __init__(
        self,
        host: AbstractMedium,
        mu_a_ex_per_m: float,
        emission_wavelength_nm: float,
        quantum_yield: float = 1.0,
        stokes_shift_nm: float = 0.0,
        wavelength_nm: float | None = None,
    ) -> None:
        if not 0.0 <= quantum_yield <= 1.0:
            raise ValueError(f"quantum_yield must be in [0, 1], got {quantum_yield}")
        if mu_a_ex_per_m < 0.0:
            raise ValueError(f"mu_a_ex_per_m must be ≥ 0, got {mu_a_ex_per_m}")
        self._host = host
        self._mu_a_ex = mu_a_ex_per_m
        self.emission_wavelength_nm = emission_wavelength_nm
        self.inelastic_yield = quantum_yield
        self.stokes_shift_nm = stokes_shift_nm
        self.wavelength_nm = wavelength_nm if wavelength_nm is not None else host.wavelength_nm

    # ── AbstractMedium interface ───────────────────────────────────────────────

    @property
    def mu_a_per_m(self) -> float:
        """Total absorption = host + fluorophore excitation absorption."""
        return self._host.mu_a_per_m + self._mu_a_ex

    @property
    def mu_s_per_m(self) -> float:
        return self._host.mu_s_per_m

    @property
    def g(self) -> float:
        return self._host.g

    @property
    def n(self) -> float:
        return self._host.n

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def mu_a_fluorophore_per_m(self) -> float:
        """Fluorophore-only absorption coefficient (m⁻¹)."""
        return self._mu_a_ex

    @property
    def quantum_yield(self) -> float:
        return self.inelastic_yield  # type: ignore[return-value]

    # ── Alternative constructor ───────────────────────────────────────────────

    @classmethod
    def from_concentration(
        cls,
        host: AbstractMedium,
        concentration_mol_per_L: float,
        molar_extinction_L_per_mol_cm: float,
        emission_wavelength_nm: float,
        quantum_yield: float = 1.0,
        stokes_shift_nm: float = 0.0,
        wavelength_nm: float | None = None,
    ) -> FluorescentMedium:
        """Construct from Beer-Lambert molar quantities.

        Parameters
        ----------
        concentration_mol_per_L : molar concentration (mol / L)
        molar_extinction_L_per_mol_cm : molar extinction coefficient ε
            (L mol⁻¹ cm⁻¹), also called molar absorptivity

        Conversion:
            μ_a (m⁻¹) = ε (L mol⁻¹ cm⁻¹) × C (mol L⁻¹) × ln(10) × 100
        """
        mu_a_ex = (
            molar_extinction_L_per_mol_cm
            * concentration_mol_per_L
            * 2.302_585  # ln(10)
            * 100.0      # cm⁻¹ → m⁻¹
        )
        return cls(
            host=host,
            mu_a_ex_per_m=mu_a_ex,
            emission_wavelength_nm=emission_wavelength_nm,
            quantum_yield=quantum_yield,
            stokes_shift_nm=stokes_shift_nm,
            wavelength_nm=wavelength_nm,
        )
