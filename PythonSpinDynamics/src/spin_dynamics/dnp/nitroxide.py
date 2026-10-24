"""Resolved-line, high-temperature nitroxide CW saturation populations.

Electron exchange follows the population-swap collision model of Bates and
Drozdoski, as described in DOI 10.1039/C5CP00935A (supporting information).
This is a solution ensemble saturation model, separate from a closed electron–
proton pair. Nitrogen Zeeman and hyperfine Boltzmann corrections are neglected:
the thermal state is an electron Boltzmann distribution times uniform nitrogen.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import array
from .nitroxide_relaxation import NitroxideRelaxationMatrix
from .overhauser import (
    BOLTZMANN_J_PER_K,
    ELECTRON_GAMMA_HZ_PER_T,
    PLANCK_J_S,
)


def _scalar(value, name, *, positive=False):
    result = array(value, name, positive=positive, nonnegative=not positive)
    if result.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(result)


def _line_parameter(value, name, n):
    result = np.asarray(value, dtype=float)
    if result.shape not in ((), (n,)) or np.any(np.isnan(result) | (result <= 0)):
        raise ValueError(
            f"{name} must be positive scalar or length {n}; +inf disables that bath"
        )
    if result.ndim:
        result = result.copy()
        result.setflags(write=False)
        return result
    return float(result)


@dataclass(frozen=True)
class NitroxideSaturationResult:
    """CW populations (..., nitrogen m, electron +/-), and total saturation.

    Electron polarization is p_minus - p_plus; line polarizations are
    conditional on nitrogen m. Total saturation uses all lines, not only the
    irradiated line. Nitrogen populations sum over the final electron axis.
    In full mode electron_coherences is rho[m,+;m,-], t2e_seconds has a final
    line axis (inverse diagonal R2 before exchange), and microwave transition
    rates are diagonal-line reference values rather than independent channels.
    """

    nitrogen_m: np.ndarray
    populations: np.ndarray
    line_polarizations: np.ndarray
    line_saturation: np.ndarray
    saturation_factor: np.ndarray
    thermal_electron_polarization: np.ndarray
    exchange_rate_per_second: np.ndarray
    t2e_seconds: np.ndarray
    microwave_transition_rates_per_second: np.ndarray
    electron_coherences: np.ndarray | None = None

    @property
    def nitrogen_populations(self):
        """Return the probability in each nitrogen hyperfine manifold."""
        return self.populations.sum(axis=-1)

    @property
    def electron_polarization(self):
        """Return ensemble p_minus - p_plus, summed over nitrogen states."""
        return np.sum(self.populations[..., 1] - self.populations[..., 0], axis=-1)


@dataclass(frozen=True)
class NitroxideSaturationModel:
    """CW electron/nitrogen populations for resolved 14N or 15N nitroxides.

    Supply intrinsic T1e/T2e as scalars or vectors in ascending nitrogen m.
    Positive infinity disables that independent bath. Hyperfine splitting is Hz. The
    optional T2 broadening coefficient adds coefficient*C to 1/T2e, independently
    of exchange; it is not automatically inferred from the collision rate.
    C is mol/L, exchange coefficient is M^-1 s^-1, and k_ex=coefficient*C is
    the decay rate of electron–nitrogen population correlations in a random
    electron-swap collision ensemble. Other exchange conventions can differ.

    Nitrogen relaxation is an adjacent-m symmetric transition rate w_N, s^-1
    per direction (not an asserted nitrogen T1). The electron bath has common
    polarization tanh(h gamma_e B / 2 k T) on all lines. Microwave rates are
    W_m=Omega1^2*T2/[2*(1+Delta_m^2*T2^2)], with Delta_m=Delta+2*pi*A*m.
    The default population path eliminates independent line coherences and
    retains its original scalar-rate behavior. Setting coherence_transfer=True,
    nitrogen_relaxation_matrix_per_second or relaxation_matrix selects a full
    generalized Bloch solve including transverse exchange transfer and all
    nitrogen population feedback blocks. Nitrogen matrix inputs are symmetric
    loss matrices, with nonpositive off-diagonals and zero sums. Full matrix
    inputs are additive; set T1=T2=inf to use only a supplied relaxation matrix.
    The full path uses first-order high-temperature deviations and linearized
    exchange, not finite-temperature detailed balance. Coherent transients,
    general anisotropic spectra, heating and concentration-dependent motion
    remain outside this CW model.
    """

    hyperfine_hz: float = 45e6
    electron_t1_seconds: float | np.ndarray = 0.5e-6
    electron_t2_seconds: float | np.ndarray = 0.1e-6
    nitrogen_transition_per_second: float = 0.0
    exchange_coefficient_per_molar_second: float = 0.0
    transverse_broadening_per_molar_second: float = 0.0
    nitrogen_spin: float = 1.0
    nitrogen_relaxation_matrix_per_second: np.ndarray | None = None
    relaxation_matrix: NitroxideRelaxationMatrix | None = None
    coherence_transfer: bool = False

    def __post_init__(self):
        if self.nitrogen_spin not in (0.5, 1.0):
            raise ValueError("nitrogen_spin must be 1 (14N) or 0.5 (15N)")
        n = len(self.nitrogen_m)
        for name in ("hyperfine_hz",):
            object.__setattr__(
                self, name, _scalar(getattr(self, name), name, positive=True)
            )
        for name in ("electron_t1_seconds", "electron_t2_seconds"):
            object.__setattr__(
                self, name, _line_parameter(getattr(self, name), name, n)
            )
        for name in (
            "nitrogen_transition_per_second",
            "exchange_coefficient_per_molar_second",
            "transverse_broadening_per_molar_second",
        ):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if np.any(
            np.asarray(self.electron_t2_seconds)
            > 2 * np.asarray(self.electron_t1_seconds)
        ):
            raise ValueError("electron_t2_seconds must be <= 2*T1e")
        if self.nitrogen_relaxation_matrix_per_second is not None:
            matrix = array(
                self.nitrogen_relaxation_matrix_per_second,
                "nitrogen_relaxation_matrix_per_second",
            )
            if matrix.shape != (n, n):
                raise ValueError(
                    "nitrogen relaxation matrix must have shape (2I+1,2I+1)"
                )
            # Validate the loss convention by embedding the spin-independent bath.
            NitroxideRelaxationMatrix(np.kron(matrix, np.eye(2)), matrix)
            matrix = matrix.copy()
            matrix.setflags(write=False)
            object.__setattr__(self, "nitrogen_relaxation_matrix_per_second", matrix)
        if self.relaxation_matrix is not None:
            if (
                not isinstance(self.relaxation_matrix, NitroxideRelaxationMatrix)
                or len(self.relaxation_matrix.transverse_per_second) != n
            ):
                raise ValueError("relaxation_matrix must match the nitrogen spin")

    def _nitrogen_loss(self):
        n = len(self.nitrogen_m)
        result = np.zeros((n, n))
        for i in range(n - 1):
            result[i : i + 2, i : i + 2] += (
                self.nitrogen_transition_per_second * np.array([[1, -1], [-1, 1]])
            )
        if self.nitrogen_relaxation_matrix_per_second is not None:
            result += self.nitrogen_relaxation_matrix_per_second
        return result

    @property
    def uses_full_matrix(self):
        """Whether CW solves retain transverse transfer and nuclear feedback."""
        return (
            self.coherence_transfer
            or self.relaxation_matrix is not None
            or self.nitrogen_relaxation_matrix_per_second is not None
        )

    def relaxation_blocks(self):
        """Return the total population and transverse loss blocks before exchange.

        Independent scalar/line T1/T2 baths, scalar/matrix nitrogen relaxation
        and a supplied full relaxation matrix are additive. Set T1=T2=+inf
        to use only the supplied full matrix and avoid double counting.
        """
        n = len(self.nitrogen_m)
        population = np.kron(self._nitrogen_loss(), np.eye(2))
        r1 = np.broadcast_to(1 / np.asarray(self.electron_t1_seconds), (n,))
        for i in range(n):
            population[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] += (
                r1[i] / 2 * np.array([[1, -1], [-1, 1]])
            )
        transverse = self._nitrogen_loss() + np.diag(
            np.broadcast_to(1 / np.asarray(self.electron_t2_seconds), (n,))
        )
        if self.relaxation_matrix is not None:
            population += self.relaxation_matrix.population_per_second
            transverse += self.relaxation_matrix.transverse_per_second
        return NitroxideRelaxationMatrix(population, transverse)

    @property
    def nitrogen_m(self):
        """Return ascending nitrogen projections labeling the hyperfine lines."""
        return np.arange(-self.nitrogen_spin, self.nitrogen_spin + 1)

    def _parameters(
        self,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s,
        temperature_kelvin,
    ):
        field, concentration, omega, delta, temperature = np.broadcast_arrays(
            array(field_tesla, "field_tesla", positive=True),
            array(concentration_molar, "concentration_molar", nonnegative=True),
            array(nutation_rad_per_s, "nutation_rad_per_s", nonnegative=True),
            array(detuning_rad_per_s, "detuning_rad_per_s"),
            array(temperature_kelvin, "temperature_kelvin", positive=True),
        )
        argument = (
            PLANCK_J_S
            * ELECTRON_GAMMA_HZ_PER_T
            * field
            / (2 * BOLTZMANN_J_PER_K * temperature)
        )
        if np.any(argument > 0.1):
            raise ValueError(
                "resolved-line model requires high temperature (electron argument <= 0.1)"
            )
        if np.any(
            ELECTRON_GAMMA_HZ_PER_T * field
            < 10 * self.hyperfine_hz * self.nitrogen_spin
        ):
            raise ValueError("field is too low for the secular resolved-line model")
        with np.errstate(over="ignore", invalid="ignore"):
            exchange = self.exchange_coefficient_per_molar_second * concentration
            r2 = np.broadcast_to(
                1 / np.asarray(self.electron_t2_seconds), (len(self.nitrogen_m),)
            )
            if self.uses_full_matrix:
                r2 = self.relaxation_blocks().transverse_per_second.diagonal()
            t2_lines = 1 / (
                r2
                + self.transverse_broadening_per_molar_second * concentration[..., None]
            )
            t2 = (
                t2_lines
                if self.uses_full_matrix or np.ndim(self.electron_t2_seconds)
                else t2_lines[..., 0]
            )
            offsets = delta[..., None] + 2 * np.pi * self.hyperfine_hz * self.nitrogen_m
            microwave = (
                0.5
                * (omega[..., None] * np.sqrt(t2_lines)) ** 2
                / (1 + (offsets * t2_lines) ** 2)
            )
        if not (
            np.isfinite(exchange).all()
            and np.isfinite(microwave).all()
            and np.all(t2 > 0)
        ):
            raise ValueError("parameters exceed finite numerical rate range")
        return np.tanh(argument), exchange, t2, microwave

    def steady_state(
        self,
        *,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s=0.0,
        temperature_kelvin=298.15,
    ):
        """Solve CW balance about a uniform-nitrogen thermal reference.

        Inputs broadcast; the final axes of populations are (2I+1, 2). At zero
        nitrogen relaxation, uniform nitrogen is a specified invariant sector,
        not a claim of a unique stationary state over all nitrogen marginals.
        The exact collision term k*(p_N outer p_e - p) reduces to a linear
        line-polarization system in this sector. Full matrices can redistribute
        nitrogen populations through the R0/U/V blocks, and solve both electron
        quadratures. No population clipping is used.
        """
        thermal, exchange, t2, microwave = self._parameters(
            field_tesla,
            concentration_molar,
            nutation_rad_per_s,
            detuning_rad_per_s,
            temperature_kelvin,
        )
        if self.uses_full_matrix:
            return self._matrix_steady_state(
                thermal,
                exchange,
                t2,
                microwave,
                concentration_molar,
                nutation_rad_per_s,
                detuning_rad_per_s,
            )
        n = len(self.nitrogen_m)
        matrix = (
            self._nitrogen_loss()
            + exchange[..., None, None] * (np.eye(n) - np.ones((n, n)) / n)
            + (1 / np.asarray(self.electron_t1_seconds) + 2 * microwave)[..., :, None]
            * np.eye(n)
        )
        if not np.isfinite(matrix).all() or np.any(np.linalg.cond(matrix) > 1e12):
            raise ValueError(
                "population balance is numerically unresolved at these rate ratios"
            )
        saturation = np.linalg.solve(matrix, (2 * microwave)[..., None])[..., 0]
        polarization = (1 - saturation) * thermal[..., None]
        populations = np.stack((1 - polarization, 1 + polarization), axis=-1) / (2 * n)
        return NitroxideSaturationResult(
            self.nitrogen_m,
            populations,
            polarization,
            saturation,
            np.mean(saturation, axis=-1),
            thermal,
            exchange,
            t2,
            microwave,
        )

    def _matrix_system(self, exchange, concentration, omega, detuning):
        n = len(self.nitrogen_m)
        blocks = self.relaxation_blocks()
        r1, u, v, r0 = blocks.longitudinal_blocks()
        # Only decaying nuclear modes are unknown; conserved nitrogen sectors
        # retain their uniform thermal values, including zero-rate degeneracies.
        values, vectors = np.linalg.eigh(r0)
        q = vectors[
            :,
            values
            > max(np.finfo(float).tiny, np.max(np.abs(values)))
            * 64
            * np.finfo(float).eps,
        ]
        count = 3 * n + q.shape[1]
        exchange, concentration, omega, detuning = np.broadcast_arrays(
            exchange, concentration, omega, detuning
        )
        matrix = np.zeros(exchange.shape + (count, count))
        mixing = np.eye(n) - np.ones((n, n)) / n
        r2 = (
            blocks.transverse_per_second
            + exchange[..., None, None] * mixing
            + (self.transverse_broadening_per_molar_second * concentration)[
                ..., None, None
            ]
            * np.eye(n)
        )
        delta = detuning[..., None] + 2 * np.pi * self.hyperfine_hz * self.nitrogen_m
        diagonal_delta = delta[..., :, None] * np.eye(n)
        identity_drive = omega[..., None, None] * np.eye(n)
        matrix[..., :n, :n] = r2
        matrix[..., n : 2 * n, n : 2 * n] = r2
        matrix[..., :n, n : 2 * n] = diagonal_delta
        matrix[..., n : 2 * n, :n] = -diagonal_delta
        matrix[..., n : 2 * n, 2 * n : 3 * n] = -identity_drive
        matrix[..., 2 * n : 3 * n, n : 2 * n] = identity_drive
        matrix[..., 2 * n : 3 * n, 2 * n : 3 * n] = (
            r1 + exchange[..., None, None] * mixing
        )
        matrix[..., 2 * n : 3 * n, 3 * n :] = u @ q
        matrix[..., 3 * n :, 2 * n : 3 * n] = q.T @ v
        matrix[..., 3 * n :, 3 * n :] = q.T @ r0 @ q
        rhs = np.zeros(exchange.shape + (count,))
        rhs[..., n : 2 * n] = omega[..., None] / n
        return matrix, rhs, q

    def _matrix_steady_state(
        self, thermal, exchange, t2, microwave, concentration, omega, detuning
    ):
        n = len(self.nitrogen_m)
        matrix, rhs, q = self._matrix_system(exchange, concentration, omega, detuning)
        if not np.isfinite(matrix).all() or np.any(np.linalg.cond(matrix) > 1e12):
            raise ValueError(
                "full relaxation balance is nonunique or numerically unresolved"
            )
        solution = np.linalg.solve(matrix, rhs[..., None])[..., 0]
        d = thermal[..., None] * (1 / n + solution[..., 2 * n : 3 * n])
        nitrogen = 1 / n + thermal[..., None] * np.einsum(
            "ij,...j->...i", q, solution[..., 3 * n :]
        )
        populations = np.stack((nitrogen - d, nitrogen + d), axis=-1) / 2
        coherences = (
            thermal[..., None] * (solution[..., :n] - 1j * solution[..., n : 2 * n]) / 2
        )
        if np.min(populations) < -1e-12 or np.any(
            np.abs(coherences) ** 2 > np.prod(populations, axis=-1) + 1e-12
        ):
            raise ValueError(
                "full relaxation matrices produce a nonphysical steady state"
            )
        polarization = d / nitrogen
        line_saturation = -solution[..., 2 * n : 3 * n] * n
        return NitroxideSaturationResult(
            self.nitrogen_m,
            populations,
            polarization,
            1 - polarization / thermal[..., None],
            np.mean(line_saturation, axis=-1),
            thermal,
            exchange,
            t2,
            microwave,
            coherences,
        )

    def population_derivative(
        self,
        populations,
        *,
        field_tesla,
        concentration_molar,
        nutation_rad_per_s,
        detuning_rad_per_s=0.0,
        temperature_kelvin=298.15,
    ):
        """Evaluate population derivatives with adiabatic electron coherences.

        Accept a single normalized nonnegative (2I+1, 2) population array and
        scalar experimental parameters. Exchange alone preserves both nitrogen
        marginals and total electron magnetization for arbitrary populations.
        In full-matrix mode coherences are eliminated as a coupled block and
        exchange is linearized to first order in thermal polarization, matching
        the CW solve. This method is not coherent transient propagation.
        """
        p = array(populations, "populations", nonnegative=True)
        if p.shape != (len(self.nitrogen_m), 2) or not np.isclose(
            p.sum(), 1, atol=1e-12, rtol=0
        ):
            raise ValueError("populations must be normalized with shape (2I+1, 2)")
        thermal, exchange, _, microwave = self._parameters(
            field_tesla,
            concentration_molar,
            nutation_rad_per_s,
            detuning_rad_per_s,
            temperature_kelvin,
        )
        if thermal.ndim:
            raise ValueError(
                "population_derivative requires scalar experimental parameters"
            )
        if self.uses_full_matrix:
            return self._matrix_population_derivative(
                p,
                thermal,
                exchange,
                concentration_molar,
                nutation_rad_per_s,
                detuning_rad_per_s,
            )
        flux = ((1 + thermal) * p[:, 0] - (1 - thermal) * p[:, 1]) / (
            2 * np.asarray(self.electron_t1_seconds)
        )
        flux += microwave * (p[:, 0] - p[:, 1])
        derivative = np.stack((-flux, flux), axis=-1)
        for index in range(len(p) - 1):
            flow = self.nitrogen_transition_per_second * (p[index + 1] - p[index])
            derivative[index] += flow
            derivative[index + 1] -= flow
        # Keep the pair-collision loss proportional to total probability, too.
        # It equals p on the normalized simplex and prevents numerical mass
        # errors from being amplified during intermediate integration stages.
        derivative += exchange * (
            p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :] - p * p.sum()
        )
        return derivative

    def _matrix_population_derivative(
        self, p, thermal, exchange, concentration, omega, detuning
    ):
        n = len(p)
        blocks = self.relaxation_blocks()
        reference = np.tile([(1 - thermal) / (2 * n), (1 + thermal) / (2 * n)], (n, 1))
        derivative = (-blocks.population_per_second @ (p - reference).ravel()).reshape(
            n, 2
        )
        mixing = np.eye(n) - np.ones((n, n)) / n
        r2 = (
            blocks.transverse_per_second
            + exchange * mixing
            + self.transverse_broadening_per_molar_second * concentration * np.eye(n)
        )
        delta = np.diag(detuning + 2 * np.pi * self.hyperfine_hz * self.nitrogen_m)
        coherence_matrix = np.block([[r2, delta], [-delta, r2]])
        rhs = np.r_[np.zeros(n), omega * (p[:, 1] - p[:, 0])]
        xy = np.linalg.solve(coherence_matrix, rhs)
        dloss = exchange * mixing @ (p[:, 1] - p[:, 0])
        flow = (omega * xy[n:] + dloss) / 2
        derivative[:, 0] += flow
        derivative[:, 1] -= flow
        return derivative
