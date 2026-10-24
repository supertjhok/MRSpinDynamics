"""Thermal, driven Lindblad reference for one S=1/2 electron and I=1/2 nucleus.

H_lab/hbar=we*Sz-wn*Iz+Azz*Sz*Iz. The returned density matrices use the
electron and nuclear rotating frames; H_rot=detuning*Sz+Azz*Sz*Iz+drive.
Thermal jumps always use laboratory energy gaps, never rotating-frame gaps.
This is a 1:1 spin pair, not an automatic dilute-solution reservoir mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._validation import array
from .motion import RotationalDipolarModel, ScalarContactModel
from .overhauser import (
    BOLTZMANN_J_PER_K,
    ELECTRON_GAMMA_HZ_PER_T,
    MU0_OVER_4PI,
    PLANCK_J_S,
    PROTON_GAMMA_HZ_PER_T,
    OverhauserRates,
    _gammas,
)


def _number(value, name, *, positive=False):
    value = array(value, name, positive=positive, nonnegative=not positive)
    if value.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(value)


def _spin_operators():
    single = (
        np.array([[0, 1], [1, 0]]) / 2,
        np.array([[0, -1j], [1j, 0]]) / 2,
        np.diag([0.5, -0.5]),
    )
    return (
        tuple(np.kron(op, np.eye(2)) for op in single),
        tuple(np.kron(np.eye(2), op) for op in single),
    )


def _dissipator(jump):
    product = jump.conj().T @ jump
    return np.kron(jump.conj(), jump) - 0.5 * (
        np.kron(np.eye(4), product) + np.kron(product.T, np.eye(4))
    )


def _affine_generator(generator):
    """Represent rho=I/4+sum(x_a*B_a) in orthonormal traceless Hermitian B_a.

    Integrating the real coordinates preserves trace and Hermiticity by
    construction, even across stiff electron/nuclear time scales. This is a
    change of coordinates, not a projection, clipping or renormalization.
    """
    basis = []
    for row in range(4):
        for col in range(row + 1, 4):
            edge = np.zeros((4, 4), dtype=complex)
            edge[row, col] = 1
            basis.extend(
                [(edge + edge.T) / np.sqrt(2), -1j * (edge - edge.T) / np.sqrt(2)]
            )
    for count in range(1, 4):
        diagonal = np.zeros(4)
        diagonal[:count], diagonal[count] = 1, -count
        basis.append(np.diag(diagonal / np.sqrt(count * (count + 1))).astype(complex))
    basis = np.asarray(basis)
    transform = np.column_stack([op.reshape(-1, order="F") for op in basis])
    matrix = (transform.conj().T @ generator @ transform).real
    source = (
        transform.conj().T @ generator @ (np.eye(4) / 4).reshape(-1, order="F")
    ).real
    return matrix, source, basis


@dataclass(frozen=True)
class ElectronNuclearTransitionRates:
    """Infinite-temperature directional transition rates W in s^-1 for one pair.

    w0: S+ I- (ZQ), w2: S+ I+ (DQ), w1n: 2*Sz*I+, w1e: 2*S+*Iz,
    and their reverses. Conditional SQ operator signs preserve the stochastic
    dipolar coupling's coherence structure. Intrinsic T1 baths are separate.
    Nuclear R_rho=w0+2*w1n+w2, R_sigma=w2-w0; these two sums alone do not
    uniquely identify w0/w1n/w2. Rates must not be negative.
    """

    zero_quantum_per_second: float = 0.0
    double_quantum_per_second: float = 0.0
    nuclear_single_quantum_per_second: float = 0.0
    electron_single_quantum_per_second: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _number(getattr(self, name), name))

    @classmethod
    def from_rotational_model(
        cls,
        model: RotationalDipolarModel,
        field_tesla,
        *,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ):
        """Resolve one occupied rotational pair into its four transition channels.

        A fractional/ensemble pair_count cannot define a closed 1:1 pair; use
        pair_count=1 here. Do not insert bulk FFHS nuclear rates into this model.
        """
        if not isinstance(model, RotationalDipolarModel):
            raise TypeError("model must be a RotationalDipolarModel")
        if model.pair_count != 1:
            raise ValueError("closed-pair dynamics requires rotational pair_count=1")
        bfield = _number(field_tesla, "field_tesla", positive=True)
        gn, ge = _gammas(gamma_n_hz_per_t, gamma_e_hz_per_t)
        wn, we = 2 * np.pi * gn * bfield, 2 * np.pi * ge * bfield
        b = (
            MU0_OVER_4PI
            * PLANCK_J_S
            / (2 * np.pi)
            * (2 * np.pi * gn)
            * (2 * np.pi * ge)
            / model.distance_m**3
        )
        a = b * b / 10
        return cls(
            float(a * model.spectral_density(we + wn)),
            float(6 * a * model.spectral_density(we - wn)),
            float(1.5 * a * model.spectral_density(wn)),
            float(1.5 * a * model.spectral_density(we)),
        )

    @classmethod
    def from_scalar_model(
        cls,
        model: ScalarContactModel,
        field_tesla,
        *,
        gamma_n_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        gamma_e_hz_per_t=ELECTRON_GAMMA_HZ_PER_T,
    ):
        """Use a pair's effective scalar autocovariance as the ZQ channel."""
        if not isinstance(model, ScalarContactModel):
            raise TypeError("model must be a ScalarContactModel")
        _number(field_tesla, "field_tesla", positive=True)
        rates = model.relaxation_rates(
            field_tesla,
            gamma_n_hz_per_t=gamma_n_hz_per_t,
            gamma_e_hz_per_t=gamma_e_hz_per_t,
        )
        return cls(zero_quantum_per_second=float(rates.self_relaxation_per_second))

    def nuclear_cw_rates(self, *, background_r1_per_second=0.0):
        """Return high-temperature nuclear rates for a CW limiting comparison."""
        return OverhauserRates(
            background_r1_per_second,
            self.zero_quantum_per_second
            + 2 * self.nuclear_single_quantum_per_second
            + self.double_quantum_per_second,
            self.double_quantum_per_second - self.zero_quantum_per_second,
            "explicit_pair_transition_rates",
        )


@dataclass(frozen=True)
class MicrowaveDrive:
    """Coherent rotating-frame drive; nutation and detuning are angular frequencies.

    H_drive=omega1*(cos(phase)*Sx+sin(phase)*Sy); detuning=we-omega_mw.
    Nutation is the co-rotating component, not a linearly polarized lab amplitude.
    """

    nutation_rad_per_s: float = 0.0
    detuning_rad_per_s: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "nutation_rad_per_s",
            _number(self.nutation_rad_per_s, "nutation_rad_per_s"),
        )
        for name in ("detuning_rad_per_s", "phase_rad"):
            value = array(getattr(self, name), name)
            if value.ndim:
                raise ValueError(f"{name} must be scalar")
            object.__setattr__(self, name, float(value))

    def off(self):
        """Turn off nutation while retaining this rotating-frame reference."""
        return MicrowaveDrive(0.0, self.detuning_rad_per_s, self.phase_rad)


@dataclass(frozen=True)
class CoupledSteadyState:
    """Normalized density matrix and magnetic polarizations for a driven pair."""

    density: np.ndarray
    electron_polarization: float
    nuclear_polarization: float
    electron_saturation: float
    nuclear_enhancement: float
    residual_per_second: float


@dataclass(frozen=True)
class CoupledDynamicsResult:
    """Time-leading densities in electron-first basis (++,+-,-+,--).

    P_e=-2<Sz> and P_n=+2<Iz> are positive at ordinary Zeeman thermal equilibrium.
    Density matrices are in the doubly rotating frame. They are not independently
    normalized deviation states; thermal polarizations retain their real scales.
    """

    times_seconds: np.ndarray
    density_matrices: np.ndarray
    electron_polarization: np.ndarray
    nuclear_polarization: np.ndarray
    thermal_electron_polarization: float
    thermal_nuclear_polarization: float
    drive: MicrowaveDrive


@dataclass(frozen=True)
class CoupledElectronNuclearModel:
    """One electron–nucleus thermal Lindblad model with secular static Azz.

    T1 values specify intrinsic lattice baths, excluding explicit cross channels.
    Dephasing rates are pure dephasing, so intrinsic 1/T2=1/(2*T1)+gamma_phi.
    Each thermal edge keeps the specified infinite-temperature directional mean
    W: k_up=2W/(1+exp(beta*gap)), k_down=2W/(1+exp(-beta*gap)).
    This guarantees Gibbs recovery for the laboratory secular Hamiltonian.
    No general anisotropic/mixed-state hyperfine or nitrogen/exchange model is
    implied. Transition rates are fixed inputs for the chosen field/temperature.
    """

    field_tesla: float
    temperature_kelvin: float = 298.15
    electron_t1_seconds: float = 1e-4
    nuclear_t1_seconds: float = 2.5
    electron_dephasing_per_second: float = 2e4
    nuclear_dephasing_per_second: float = 0.0
    secular_hyperfine_hz: float = 0.0
    transitions: ElectronNuclearTransitionRates = field(
        default_factory=ElectronNuclearTransitionRates
    )
    gamma_n_hz_per_t: float = PROTON_GAMMA_HZ_PER_T
    gamma_e_hz_per_t: float = ELECTRON_GAMMA_HZ_PER_T

    def __post_init__(self):
        for name in ("field_tesla", "temperature_kelvin"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )
        for name in ("electron_t1_seconds", "nuclear_t1_seconds"):
            value = float(getattr(self, name))
            if np.isnan(value) or value <= 0:
                raise ValueError(f"{name} must be positive or +infinity")
            object.__setattr__(self, name, value)
        for name in ("electron_dephasing_per_second", "nuclear_dephasing_per_second"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        hyperfine = array(self.secular_hyperfine_hz, "secular_hyperfine_hz")
        if hyperfine.ndim:
            raise ValueError("secular_hyperfine_hz must be scalar")
        object.__setattr__(self, "secular_hyperfine_hz", float(hyperfine))
        gn, ge = _gammas(self.gamma_n_hz_per_t, self.gamma_e_hz_per_t)
        object.__setattr__(self, "gamma_n_hz_per_t", gn)
        object.__setattr__(self, "gamma_e_hz_per_t", ge)
        if not isinstance(self.transitions, ElectronNuclearTransitionRates):
            raise TypeError("transitions must be ElectronNuclearTransitionRates")

    def laboratory_hamiltonian(self):
        """Return the secular laboratory Hamiltonian H/hbar in rad/s."""
        electron, nucleus = _spin_operators()
        return (
            2
            * np.pi
            * self.field_tesla
            * (self.gamma_e_hz_per_t * electron[2] - self.gamma_n_hz_per_t * nucleus[2])
            + 2 * np.pi * self.secular_hyperfine_hz * electron[2] @ nucleus[2]
        )

    def thermal_density(self):
        """Return the trace-one Gibbs state of H_lab, using exact Boltzmann factors."""
        energies = np.diag(self.laboratory_hamiltonian()).real
        exponent = (
            -PLANCK_J_S
            / (2 * np.pi * BOLTZMANN_J_PER_K * self.temperature_kelvin)
            * (energies - energies.min())
        )
        populations = np.exp(exponent)
        return np.diag((populations / populations.sum()).astype(complex))

    def polarizations(self, density):
        """Return magnetic P_e=-2<Sz>, P_n=2<Iz> for a density or density stack."""
        electron, nucleus = _spin_operators()
        density = np.asarray(density, dtype=complex)
        return (
            -2 * np.einsum("...ij,ji->...", density, electron[2]).real,
            2 * np.einsum("...ij,ji->...", density, nucleus[2]).real,
        )

    def jump_operators(self):
        """Build detailed-balance jumps using laboratory gaps and grouped degeneracies.

        Degenerate edges from the same bath are one jump, preserving spectator
        coherences. Resolved hyperfine-split edges are separate secular jumps.
        The directional finite-T factors are numerically stable logistic weights.
        """
        electron, nucleus = _spin_operators()
        sp, ip = electron[0] + 1j * electron[1], nucleus[0] + 1j * nucleus[1]
        channels = (
            (sp, 0.5 / self.electron_t1_seconds),
            (ip, 0.5 / self.nuclear_t1_seconds),
            (sp @ ip.conj().T, self.transitions.zero_quantum_per_second),
            (sp @ ip, self.transitions.double_quantum_per_second),
            (2 * electron[2] @ ip, self.transitions.nuclear_single_quantum_per_second),
            (2 * sp @ nucleus[2], self.transitions.electron_single_quantum_per_second),
        )
        energies = np.diag(self.laboratory_hamiltonian()).real
        beta_hbar = PLANCK_J_S / (
            2 * np.pi * BOLTZMANN_J_PER_K * self.temperature_kelvin
        )
        jumps = []
        for operator, rate in channels:
            if rate == 0:
                continue
            for directed in (operator, operator.conj().T):
                groups = []
                for row, col in zip(*np.nonzero(directed)):
                    gap = energies[row] - energies[col]
                    group = next(
                        (
                            item
                            for item in groups
                            if np.isclose(
                                item[0], gap, rtol=64 * np.finfo(float).eps, atol=1e-10
                            )
                        ),
                        None,
                    )
                    if group is None:
                        group = (gap, np.zeros((4, 4), dtype=complex))
                        groups.append(group)
                    group[1][row, col] = directed[row, col]
                for gap, jump in groups:
                    biased_rate = 2 * rate * np.exp(-np.logaddexp(0, beta_hbar * gap))
                    if biased_rate:
                        jumps.append(np.sqrt(biased_rate) * jump)
        for rate, z in (
            (self.electron_dephasing_per_second, electron[2]),
            (self.nuclear_dephasing_per_second, nucleus[2]),
        ):
            if rate:
                jumps.append(np.sqrt(2 * rate) * z)
        return tuple(jumps)

    def liouvillian(self, drive: MicrowaveDrive | None = None):
        """Return the 16x16 generator for column-stacked rotating-frame densities."""
        drive = MicrowaveDrive() if drive is None else drive
        if not isinstance(drive, MicrowaveDrive):
            raise TypeError("drive must be MicrowaveDrive")
        electron, nucleus = _spin_operators()
        h = (
            drive.detuning_rad_per_s * electron[2]
            + 2 * np.pi * self.secular_hyperfine_hz * electron[2] @ nucleus[2]
            + drive.nutation_rad_per_s
            * (
                np.cos(drive.phase_rad) * electron[0]
                + np.sin(drive.phase_rad) * electron[1]
            )
        )
        generator = -1j * (np.kron(np.eye(4), h) - np.kron(h.T, np.eye(4)))
        for jump in self.jump_operators():
            generator += _dissipator(jump)
        return generator

    def steady_state(self, drive: MicrowaveDrive | None = None) -> CoupledSteadyState:
        """Solve the trace-constrained stationary equation; reject nonunique states."""
        generator = self.liouvillian(drive)
        matrix, source, basis = _affine_generator(generator)
        scale = np.max(np.abs(matrix))
        normalized = matrix / (scale if scale else 1)
        if np.linalg.matrix_rank(normalized) < 15:
            raise ValueError(
                "stationary state is not unique or is numerically unresolved"
            )
        coordinates = np.linalg.solve(normalized, -source / (scale if scale else 1))
        density = np.eye(4) / 4 + np.einsum("a,aij->ij", coordinates, basis)
        vector = density.reshape(-1, order="F")
        _validate_density(density)
        pe, pn = self.polarizations(density)
        pe0, pn0 = self.polarizations(self.thermal_density())
        return CoupledSteadyState(
            density,
            float(pe),
            float(pn),
            float(1 - pe / pe0) if pe0 else np.nan,
            float(pn / pn0) if pn0 else np.nan,
            float(np.linalg.norm(generator @ vector)),
        )

    def simulate(
        self,
        times_seconds,
        *,
        drive: MicrowaveDrive | None = None,
        initial_density=None,
    ) -> CoupledDynamicsResult:
        """Propagate a constant drive with robust matrix exponentials (SciPy opt extra).

        Times are elapsed since the supplied initial state and must be sorted,
        nonnegative and nonempty. To turn microwaves off, start a second run with
        drive=previous_drive.off() and the final pumped density. No clipping or renormalization
        is applied to the propagated state. The first requested time may be >0.
        Changing detuning between segments changes the electron frame; transform
        transverse coherences accordingly, or retain the same detuning with off().
        """
        from scipy.linalg import expm

        times = array(times_seconds, "times_seconds", nonnegative=True)
        if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) < 0):
            raise ValueError("times_seconds must be a non-empty sorted 1-D axis")
        initial = (
            self.thermal_density()
            if initial_density is None
            else np.asarray(initial_density, dtype=complex)
        )
        _validate_density(initial)
        drive = MicrowaveDrive() if drive is None else drive
        generator = self.liouvillian(drive)
        matrix, source, basis = _affine_generator(generator)
        augmented = np.zeros((16, 16))
        augmented[:15, :15], augmented[:15, 15] = matrix, source
        coordinates = np.einsum("aij,ji->a", basis, initial).real
        vector = np.r_[coordinates, 1.0]
        densities = np.empty((times.size, 4, 4), dtype=complex)
        cache, previous = {}, 0.0
        for index, time in enumerate(times):
            duration = float(time - previous)
            if duration:
                if duration not in cache:
                    cache[duration] = expm(augmented * duration)
                vector = cache[duration] @ vector
            densities[index] = np.eye(4) / 4 + np.einsum(
                "a,aij->ij", vector[:15], basis
            )
            previous = time
        pe, pn = self.polarizations(densities)
        pe0, pn0 = self.polarizations(self.thermal_density())
        return CoupledDynamicsResult(
            times.copy(), densities, pe, pn, float(pe0), float(pn0), drive
        )


def _validate_density(density):
    if density.shape != (4, 4) or not np.isfinite(density).all():
        raise ValueError("density must be a finite 4x4 matrix")
    if not np.allclose(density, density.conj().T, rtol=0, atol=1e-11):
        raise ValueError("density must be Hermitian")
    if not np.isclose(np.trace(density), 1, rtol=0, atol=1e-11):
        raise ValueError("density must have trace one")
    if np.linalg.eigvalsh(density).min() < -1e-11:
        raise ValueError("density must be positive semidefinite")
