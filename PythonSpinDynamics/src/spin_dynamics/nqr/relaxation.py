"""NQR relaxation helpers and backward-compatible shared-model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spin_dynamics.nqr.orientations import powder_frame_grid
from spin_dynamics.relaxation import (
    DipolarRelaxationSource,
    IsotropicLiquidMotionalAveraging,
    MotionalAveragingModel,
    NQRRelaxationLike,
    NQRRelaxationModel,
    NQRRelaxationSuperoperator,
    PhenomenologicalRelaxationModel,
    RedfieldDipolarRelaxationModel,
    RedfieldEFGRelaxationModel,
    RelaxationModelLike,
    RelaxationSuperoperator,
    RigidSolidMotionalAveraging,
    VibrationalMotionalAveraging,
    cycle_superoperator,
    dipolar_coupling_hz,
    dipolar_coupling_tensor,
    effective_decay_time,
    liouville_hamiltonian,
    liouville_superoperator,
    matrix_exponential,
    propagate_density_liouville,
    relaxation_superoperator,
    quadrupolar_tesseral_operators,
    single_spin_matrices,
)


@dataclass(frozen=True)
class ZeroFieldRedfieldEFGModel:
    """Non-diagonal zero-field Redfield model for fluctuating EFGs.

    This follows Goesweiner, Westlund, and Scharfetter, Molecular Physics 118,
    e1743888 (2020), Equations (A3a-b). Unlike a secular Lindblad bath, the
    relaxation matrix retains terms that couple Kramers-degenerate single- and
    double-quantum coherences. It is intended for NQR lineshape validation, not
    as a generally completely-positive time-domain propagator.
    """

    spin: float
    fluctuation_amplitude_hz: float
    correlation_time_seconds: float
    eta: float = 0.0
    vibration_frequency_hz: float = 0.0
    n_theta: int = 4
    n_phi: int = 8
    n_chi: int = 4
    secular_tolerance_rad_per_s: float = 1.0e-3

    def __post_init__(self) -> None:
        spin = float(self.spin)
        amplitude = float(self.fluctuation_amplitude_hz)
        tau = float(self.correlation_time_seconds)
        eta = float(self.eta)
        vibration = float(self.vibration_frequency_hz)
        tolerance = float(self.secular_tolerance_rad_per_s)
        if spin <= 0.0 or round(2.0 * spin) % 2 != 1:
            raise ValueError("spin must be a positive half-integer")
        if not np.isfinite(amplitude) or amplitude < 0.0:
            raise ValueError("fluctuation_amplitude_hz must be non-negative")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("correlation_time_seconds must be positive")
        if not np.isfinite(eta) or eta < 0.0 or eta > 1.0:
            raise ValueError("eta must be in [0, 1]")
        if not np.isfinite(vibration) or vibration < 0.0:
            raise ValueError("vibration_frequency_hz must be non-negative")
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("secular_tolerance_rad_per_s must be non-negative")
        for value, name in (
            (self.n_theta, "n_theta"),
            (self.n_phi, "n_phi"),
            (self.n_chi, "n_chi"),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "fluctuation_amplitude_hz", amplitude)
        object.__setattr__(self, "correlation_time_seconds", tau)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "vibration_frequency_hz", vibration)
        object.__setattr__(self, "secular_tolerance_rad_per_s", tolerance)

    def spectral_density(self, angular_frequency_rad_per_s: float) -> float:
        """Return the shifted Lorentzian of paper Equations (16-17)."""

        # The paper evaluates the real classical spectral density at transition
        # magnitudes. Using ``abs(omega)`` supplies the required even extension
        # to negative Bohr frequencies while retaining its shifted-Lorentzian
        # convention for positive NQR frequencies.
        detuning = (
            abs(float(angular_frequency_rad_per_s))
            - 2.0 * np.pi * self.vibration_frequency_hz
        )
        tau = self.correlation_time_seconds
        return tau / (1.0 + (detuning * tau) ** 2)

    def _fluctuation_operators(
        self, eigenvectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ops = single_spin_matrices(self.spin)
        spin_squared = self.spin * (self.spin + 1.0) * ops.identity
        scale = (
            2.0
            * np.pi
            * self.fluctuation_amplitude_hz
            / (4.0 * self.spin * (2.0 * self.spin - 1.0))
        )
        frames = powder_frame_grid(self.n_theta, self.n_phi, self.n_chi)
        spin_axes = (ops.ix, ops.iy, ops.iz)
        operators: list[np.ndarray] = []
        for frame in frames:
            ix = sum(float(frame.x[k]) * spin_axes[k] for k in range(3))
            iy = sum(float(frame.y[k]) * spin_axes[k] for k in range(3))
            iz = sum(float(frame.z[k]) * spin_axes[k] for k in range(3))
            operator = scale * (
                3.0 * (iz @ iz)
                - spin_squared
                + self.eta * (ix @ ix - iy @ iy)
            )
            operators.append(eigenvectors.conj().T @ operator @ eigenvectors)
        weights = np.asarray([frame.weight for frame in frames])
        return np.asarray(operators), weights

    def _energy_basis_data(
        self, hamiltonian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        dimension = int(round(2.0 * self.spin + 1.0))
        if hamiltonian.shape != (dimension, dimension):
            raise ValueError("hamiltonian dimension does not match spin")
        energies, vectors = np.linalg.eigh(
            0.5 * (hamiltonian + hamiltonian.conj().T)
        )
        # ``eigh`` may return an arbitrary basis inside every Kramers doublet.
        # Equation (A3a) labels the partners by the sign of m, so choose the
        # deterministic basis that diagonalizes Iz within each degenerate pair.
        iz = single_spin_matrices(self.spin).iz
        for start in range(0, dimension, 2):
            pair = vectors[:, start : start + 2]
            projected_iz = pair.conj().T @ iz @ pair
            _, rotation = np.linalg.eigh(
                0.5 * (projected_iz + projected_iz.conj().T)
            )
            vectors[:, start : start + 2] = pair @ rotation
        operators, weights = self._fluctuation_operators(vectors)
        covariance = np.einsum(
            "kab,kdc,k->abdc",
            operators,
            operators,
            weights,
            optimize=True,
        )
        return energies, vectors, covariance

    def relaxation_matrix(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Return the paper's non-diagonal Redfield matrix ``Gamma``."""

        energies, _, covariance = self._energy_basis_data(hamiltonian)
        dimension = energies.size
        gaps = energies[:, None] - energies[None, :]
        result = np.zeros((dimension**2, dimension**2), dtype=np.complex128)

        def j(a: int, ap: int, b: int, bp: int, omega: float) -> complex:
            return covariance[a, ap, bp, b] * self.spectral_density(omega)

        partner = np.arange(dimension) ^ 1
        for a in range(dimension):
            for ap in range(dimension):
                row = a + ap * dimension
                for b in range(dimension):
                    for bp in range(dimension):
                        if (
                            abs(gaps[a, ap] - gaps[b, bp])
                            > self.secular_tolerance_rad_per_s
                        ):
                            continue
                        value = -j(a, b, ap, bp, gaps[a, b])
                        value -= j(a, b, ap, bp, gaps[bp, ap])
                        first_sum = sum(
                            j(g, b, g, a, gaps[g, b])
                            for g in range(dimension)
                        )
                        second_sum = sum(
                            j(g, ap, g, bp, gaps[bp, g])
                            for g in range(dimension)
                        )
                        if ap == bp or ap == partner[bp]:
                            value += first_sum
                        if a == b or a == partner[b]:
                            value += second_sum
                        result[row, b + bp * dimension] = value
        return result

    def transition_hwhm_per_second(
        self,
        hamiltonian: np.ndarray,
        detection_operator: np.ndarray,
        transition_angular_frequency_rad_per_s: float,
        *,
        frequency_tolerance_rad_per_s: float = 1.0,
    ) -> float:
        """Return the coupled-coherence absorption HWHM for one NQR line."""

        energies, vectors, _ = self._energy_basis_data(hamiltonian)
        gamma_matrix = self.relaxation_matrix(hamiltonian)
        dimension = energies.size
        gaps = energies[:, None] - energies[None, :]
        frequency_mask = np.abs(
            gaps - float(transition_angular_frequency_rad_per_s)
        ) <= float(frequency_tolerance_rad_per_s)
        # Equation (18) keeps the two same-sign coherences between adjacent
        # Kramers doublets. Only the lowest line also includes the two
        # opposite-sign double-quantum coherences; higher opposite-sign pairs
        # have nominal coherence order greater than two and are excluded even
        # though eta mixing can give them tiny Ix^2 matrix elements.
        observable_mask = np.zeros_like(frequency_mask)
        for upper in range(dimension):
            for lower in range(dimension):
                upper_pair = upper // 2
                lower_pair = lower // 2
                if upper_pair != lower_pair + 1:
                    continue
                same_sign = upper % 2 == lower % 2
                if same_sign or lower_pair == 0:
                    observable_mask[upper, lower] = True
        mask = frequency_mask & observable_mask
        indices = np.flatnonzero(mask.reshape(-1, order="F"))
        if indices.size == 0:
            raise ValueError("no coherences match the requested transition")
        block = gamma_matrix[np.ix_(indices, indices)]
        detector = vectors.conj().T @ np.asarray(
            detection_operator, dtype=np.complex128
        ) @ vectors
        vector = detector.reshape(-1, order="F")[indices]
        if np.linalg.norm(vector) == 0.0:
            raise ValueError("detection operator does not address the transition")
        if indices.size == 4:
            # The lowest line is the coupled 4x4 single/double-coherence block
            # of Equations (A13-A17). Its experimentally visible broad mode is
            # the largest positive decay eigenvalue; using only the diagonal
            # term is precisely the approximation corrected by the paper.
            rates = np.real(np.linalg.eigvals(block))
            positive = rates[rates > 0.0]
            if positive.size == 0:
                raise ValueError("lowest-transition block has no decaying mode")
            return float(np.max(positive))
        # The higher transition blocks are diagonal in the paper's selected
        # coherence basis. Average the Kramers-equivalent diagonal pair to
        # suppress floating-point basis noise.
        rate = float(np.mean(np.real(np.diag(block))))
        if rate <= 0.0:
            raise ValueError("transition relaxation rate is non-positive")
        return rate


@dataclass(frozen=True)
class SpectralOverlapRelaxationFit:
    """Linear fit of a background rate plus spectral-overlap relaxation.

    The model is ``R2 = R_floor + R_cross * S``, where ``S`` is a
    dimensionless spectral-overlap factor normalized to one at the reference
    condition. It is useful when a Hamiltonian calculation predicts how an
    applied field separates otherwise resonant homonuclear transitions, while
    the absolute dipolar strength and other relaxation channels remain nuisance
    parameters.
    """

    floor_rate_per_second: float
    cross_relaxation_rate_per_second: float
    overlap_factors: np.ndarray
    predicted_rates_per_second: np.ndarray
    residuals_per_second: np.ndarray

    @property
    def rms_residual_per_second(self) -> float:
        """Return the unweighted RMS rate residual."""

        return float(np.sqrt(np.mean(self.residuals_per_second**2)))

    @property
    def predicted_t2_seconds(self) -> np.ndarray:
        """Return lifetimes corresponding to the fitted rates."""

        return 1.0 / self.predicted_rates_per_second


def transition_rms_linewidth_hz(
    frequency_offsets_hz: Iterable[float] | np.ndarray,
    intensities: Iterable[float] | np.ndarray,
    *,
    intrinsic_sigma_hz: float = 0.0,
) -> float:
    """Return the intensity-weighted RMS transition width.

    ``intrinsic_sigma_hz`` is an independent Gaussian standard deviation and is
    therefore added in quadrature. The frequency offsets may be measured from
    any common reference because the weighted mean is removed.
    """

    offsets = np.asarray(frequency_offsets_hz, dtype=np.float64)
    weights = np.asarray(intensities, dtype=np.float64)
    if offsets.ndim != 1 or weights.ndim != 1 or offsets.shape != weights.shape:
        raise ValueError("frequency_offsets_hz and intensities must be matching 1-D arrays")
    if offsets.size == 0:
        raise ValueError("at least one transition is required")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(weights)):
        raise ValueError("frequency offsets and intensities must be finite")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("intensities must be non-negative with a positive sum")
    sigma0 = float(intrinsic_sigma_hz)
    if not np.isfinite(sigma0) or sigma0 < 0.0:
        raise ValueError("intrinsic_sigma_hz must be non-negative and finite")

    normalized = weights / np.sum(weights)
    mean = float(np.sum(normalized * offsets))
    variance = float(np.sum(normalized * (offsets - mean) ** 2))
    return float(np.sqrt(max(variance, 0.0) + sigma0**2))


def spectral_overlap_factors(
    linewidths_hz: Iterable[float] | np.ndarray,
    *,
    reference_index: int = 0,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return normalized overlap factors from transition linewidths.

    For Lorentzian flip-flop overlap, the zero-frequency overlap scales as the
    inverse width, giving the default ``S = (width_ref / width)**1``. The
    exponent is explicit so alternative independently justified overlap laws
    can be tested without changing the fitting routine.
    """

    linewidths = np.asarray(linewidths_hz, dtype=np.float64)
    if linewidths.ndim != 1 or linewidths.size == 0:
        raise ValueError("linewidths_hz must be a non-empty 1-D array")
    if not np.all(np.isfinite(linewidths)) or np.any(linewidths <= 0.0):
        raise ValueError("linewidths_hz must be positive and finite")
    index = int(reference_index)
    if index < 0 or index >= linewidths.size:
        raise ValueError("reference_index is outside linewidths_hz")
    power = float(exponent)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("exponent must be positive and finite")
    return (linewidths[index] / linewidths) ** power


def fit_spectral_overlap_relaxation(
    t2_seconds: Iterable[float] | np.ndarray,
    overlap_factors: Iterable[float] | np.ndarray,
    *,
    rate_standard_errors_per_second: Iterable[float] | np.ndarray | None = None,
) -> SpectralOverlapRelaxationFit:
    """Fit ``R2 = R_floor + R_cross * S`` by linear least squares.

    Optional rate standard errors produce an inverse-variance weighted fit.
    The returned residuals always use physical, unweighted rate units.
    """

    lifetimes = np.asarray(t2_seconds, dtype=np.float64)
    overlap = np.asarray(overlap_factors, dtype=np.float64)
    if lifetimes.ndim != 1 or overlap.ndim != 1 or lifetimes.shape != overlap.shape:
        raise ValueError("t2_seconds and overlap_factors must be matching 1-D arrays")
    if lifetimes.size < 2:
        raise ValueError("at least two measurements are required")
    if not np.all(np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
        raise ValueError("t2_seconds must be positive and finite")
    if not np.all(np.isfinite(overlap)) or np.any(overlap < 0.0):
        raise ValueError("overlap_factors must be non-negative and finite")
    if np.ptp(overlap) == 0.0:
        raise ValueError("overlap_factors must span at least two values")

    rates = 1.0 / lifetimes
    design = np.column_stack((np.ones_like(overlap), overlap))
    if rate_standard_errors_per_second is not None:
        errors = np.asarray(rate_standard_errors_per_second, dtype=np.float64)
        if errors.shape != rates.shape or not np.all(np.isfinite(errors)):
            raise ValueError("rate standard errors must match measurements and be finite")
        if np.any(errors <= 0.0):
            raise ValueError("rate standard errors must be positive")
        scale = 1.0 / errors
        fit_design = design * scale[:, None]
        fit_rates = rates * scale
    else:
        fit_design = design
        fit_rates = rates

    coefficients, _, rank, _ = np.linalg.lstsq(fit_design, fit_rates, rcond=None)
    if rank != 2:
        raise ValueError("spectral-overlap fit is rank deficient")
    predicted = design @ coefficients
    if np.any(predicted <= 0.0):
        raise ValueError("fit predicts a non-positive relaxation rate")
    return SpectralOverlapRelaxationFit(
        floor_rate_per_second=float(coefficients[0]),
        cross_relaxation_rate_per_second=float(coefficients[1]),
        overlap_factors=overlap.copy(),
        predicted_rates_per_second=predicted,
        residuals_per_second=rates - predicted,
    )

__all__ = [
    "DipolarRelaxationSource",
    "IsotropicLiquidMotionalAveraging",
    "MotionalAveragingModel",
    "NQRRelaxationLike",
    "NQRRelaxationModel",
    "NQRRelaxationSuperoperator",
    "PhenomenologicalRelaxationModel",
    "RedfieldDipolarRelaxationModel",
    "RedfieldEFGRelaxationModel",
    "RelaxationModelLike",
    "RelaxationSuperoperator",
    "RigidSolidMotionalAveraging",
    "VibrationalMotionalAveraging",
    "ZeroFieldRedfieldEFGModel",
    "SpectralOverlapRelaxationFit",
    "cycle_superoperator",
    "dipolar_coupling_hz",
    "dipolar_coupling_tensor",
    "effective_decay_time",
    "fit_spectral_overlap_relaxation",
    "liouville_hamiltonian",
    "liouville_superoperator",
    "matrix_exponential",
    "propagate_density_liouville",
    "relaxation_superoperator",
    "quadrupolar_tesseral_operators",
    "spectral_overlap_factors",
    "transition_rms_linewidth_hz",
]
