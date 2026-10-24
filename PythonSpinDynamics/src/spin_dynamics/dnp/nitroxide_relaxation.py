"""Full electron/nitrogen relaxation blocks for resolved nitroxide CW models.

Population order is (m_N ascending, electron +/-). Matrices are relaxation
loss operators: dp/dt=-R_pop*(p-p_eq), dc/dt=-R2*c. The tumbling factory
projects the double-commutator operators of Sezer et al., PCCP 2009,
DOI 10.1039/B906719C, Appendix A1, retaining nitrogen population feedback
and electron coherence transfer. This is a high-temperature secular model.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import array
from .overhauser import ELECTRON_GAMMA_HZ_PER_T


def _spin_operators(spin):
    if spin not in (0.5, 1):
        raise ValueError("nitrogen_spin must be 0.5 or 1")
    m = np.arange(-spin, spin + 1)
    n = len(m)
    ip = np.zeros((n, n))
    for i in range(n - 1):
        ip[i + 1, i] = np.sqrt(spin * (spin + 1) - m[i] * (m[i] + 1))
    return (
        np.kron(np.eye(n), np.diag([0.5, -0.5])),
        np.kron(np.eye(n), [[0, 1], [0, 0]]),
        np.kron(np.diag(m), np.eye(2)),
        np.kron(ip, np.eye(2)),
    )


@dataclass(frozen=True)
class NitroxideRelaxationMatrix:
    """Population (2n,2n) and transverse (n,n) real symmetric loss matrices.

    R_pop must be a symmetric Markov loss operator (zero row/column sums,
    nonpositive off-diagonals). R2 must be positive semidefinite. These checks
    ensure classical population stability, not complete positivity of an
    arbitrary user-supplied pair of blocks. The tumbling factory constructs
    both blocks from one common relaxation superoperator. Arrays are copied
    and made read-only. n=2 for 15N or n=3 for 14N.
    """

    population_per_second: np.ndarray
    transverse_per_second: np.ndarray

    def __post_init__(self):
        p = array(self.population_per_second, "population_per_second")
        t = array(self.transverse_per_second, "transverse_per_second")
        if p.shape not in ((4, 4), (6, 6)) or t.shape != (len(p) // 2,) * 2:
            raise ValueError(
                "require population shape (4,4)/(6,6) and transverse shape (n,n)"
            )
        for value, name in ((p, "population"), (t, "transverse")):
            scale = max(1, np.linalg.norm(value, ord=np.inf))
            if not np.allclose(value, value.T, rtol=0, atol=1e-12 * scale):
                raise ValueError(f"{name} matrix must be symmetric")
            if np.linalg.eigvalsh(value).min() < -1e-12 * scale:
                raise ValueError(f"{name} matrix must be positive semidefinite")
        scale = max(1, np.linalg.norm(p, ord=np.inf))
        if (
            np.max(p - np.diag(np.diag(p))) > 1e-12 * scale
            or np.max(np.abs(p.sum(axis=0))) > 1e-12 * scale
        ):
            raise ValueError(
                "population matrix must have nonpositive off-diagonals and zero column sums"
            )
        for name, value in (("population_per_second", p), ("transverse_per_second", t)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @classmethod
    def from_isotropic_tumbling(
        cls,
        *,
        field_tesla,
        correlation_time_seconds,
        g_tensor,
        hyperfine_tensor_hz,
        nitrogen_spin=1,
        spin_rotation_j0_per_second=0,
        spin_rotation_jomega_per_second=0,
        additional_r1_per_second=0,
        additional_r2_per_second=0,
    ):
        """Build all nitrogen/electron blocks from aligned principal tensors.

        g is dimensionless; A is in Hz, not angular frequency or gauss. The
        rank-two correlation time is tau_D=1/(6*D_rot). g/A anisotropy and their
        cross-correlation use J_XY=c_XY*tau/(1+omega_e^2*tau^2), c_XY=Tr(X'Y')/5,
        with X=omega_e*g/g_free and Y=2*pi*A. Spin-rotation spectral densities
        at zero/electron frequency are independent supplied nonnegative inputs.
        Additional R1/R2 are independent electron rates and require R2>=R1/2.
        This does not include nitrogen quadrupolar relaxation or nonsecular
        hyperfine shifts. Zero tensors/rates are allowed.
        """
        field = array(field_tesla, "field_tesla", positive=True)
        tau = array(correlation_time_seconds, "correlation_time_seconds", positive=True)
        if field.ndim or tau.ndim:
            raise ValueError("field and correlation time must be scalar")
        g = array(g_tensor, "g_tensor", positive=True)
        a = 2 * np.pi * array(hyperfine_tensor_hz, "hyperfine_tensor_hz")
        if g.shape != (3,) or a.shape != (3,):
            raise ValueError("aligned principal tensors must each contain three values")
        rates = []
        for value, name in (
            (spin_rotation_j0_per_second, "spin_rotation_j0_per_second"),
            (spin_rotation_jomega_per_second, "spin_rotation_jomega_per_second"),
            (additional_r1_per_second, "additional_r1_per_second"),
            (additional_r2_per_second, "additional_r2_per_second"),
        ):
            rate = array(value, name, nonnegative=True)
            if rate.ndim:
                raise ValueError(f"{name} must be scalar")
            rates.append(float(rate))
        j0, jw, r1, r2 = rates
        if r2 < r1 / 2:
            raise ValueError("additional R2 must be >= additional R1/2")
        omega = 2 * np.pi * ELECTRON_GAMMA_HZ_PER_T * field
        gg = omega * (g - g.mean()) / 2.00231930436256
        aa = a - a.mean()
        cgg, cga, caa = np.dot(gg, gg) / 5, np.dot(gg, aa) / 5, np.dot(aa, aa) / 5
        attenuation = 1 / (1 + (omega * tau) ** 2)
        g0, gw = cgg * tau, cgg * tau * attenuation
        ga0, gaw = cga * tau, cga * tau * attenuation
        a0, aw = caa * tau, caa * tau * attenuation
        sz, sp, iz, ip = _spin_operators(nitrogen_spin)
        sm, im = sp.T, ip.T
        terms = []

        def add(coefficient, left, right):
            terms.append((float(coefficient), left, right))

        add(j0 + 2 * g0 / 3 + r2 - r1 / 2, sz, sz)
        for left, right in ((sm, sp), (sp, sm)):
            add(jw / 2 + gw / 4 + r1 / 4, left, right)
        for left, right in ((sz, iz @ sz), (iz @ sz, sz)):
            add(2 * ga0 / 3, left, right)
        for left, right in ((sm, iz @ sp), (iz @ sm, sp), (sp, iz @ sm), (iz @ sp, sm)):
            add(gaw / 4, left, right)
        add(2 * a0 / 3, iz @ sz, iz @ sz)
        for left, right in ((ip @ sz, im @ sz), (im @ sz, ip @ sz)):
            add(a0 / 4, left, right)
        for left, right, weight in (
            (iz @ sm, iz @ sp, 1),
            (iz @ sp, iz @ sm, 1),
            (ip @ sm, im @ sp, 1 / 6),
            (im @ sp, ip @ sm, 1 / 6),
            (ip @ sp, im @ sm, 1),
            (im @ sm, ip @ sp, 1),
        ):
            add(aw * weight / 4, left, right)

        def relax(rho):
            out = np.zeros_like(rho, dtype=complex)
            for coefficient, left, right in terms:
                inner = right @ rho - rho @ right
                out += coefficient * (left @ inner - inner @ left)
            return out

        n = len(sz) // 2
        population = np.zeros((2 * n, 2 * n))
        transverse = np.zeros((n, n))
        for j in range(2 * n):
            state = np.zeros((2 * n, 2 * n))
            state[j, j] = 1
            population[:, j] = relax(state).diagonal().real
        for j in range(n):
            state = np.zeros((2 * n, 2 * n))
            state[2 * j, 2 * j + 1] = 1
            transformed = relax(state)
            transverse[:, j] = np.array(
                [transformed[2 * i, 2 * i + 1].real for i in range(n)]
            )
        return cls(population, transverse)

    def longitudinal_blocks(self):
        """Return full (R1,U,V,R0) blocks for d=[p_minus-p_plus], q=[p_sum].

        d_dot=-R1*d-U*q and q_dot=-V*d-R0*q for deviations from the
        factorized thermal reference. The trace-zero sector of q is retained
        by the CW solver; no nuclear feedback block is discarded.
        """
        n = len(self.transverse_per_second)
        change = np.zeros((2 * n, 2 * n))
        for i in range(n):
            change[i, 2 * i : 2 * i + 2] = [-1, 1]
            change[n + i, 2 * i : 2 * i + 2] = [1, 1]
        block = change @ self.population_per_second @ change.T / 2
        return block[:n, :n], block[:n, n:], block[n:, :n], block[n:, n:]
