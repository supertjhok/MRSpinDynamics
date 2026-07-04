"""Eddy-current L/R eigenmodes of a conducting structure -> gradient pre-emphasis.

Models a thin conducting structure (e.g. a cryostat bore) as a set of coaxial
filamentary rings. A step in a gradient coil's current induces eddy currents in
the rings that decay with the structure's ``L/R`` time constants and produce a
residual gradient field at the sample -- the classic gradient eddy-current
droop. This solver builds the ring resistance ``R``, inductance ``L`` and
drive-coupling ``M`` from the geometry, eigen-decomposes the ``L^{-1}R``
dynamics, and returns the field decay as ``sum_k alpha_k exp(-t/tau_k)`` --
exactly the ``eddy_terms`` of the M5
:class:`optimal_control.control_response.GradientDriverResponse`.

So coil + shield geometry now *predicts* the pre-emphasis terms that were hand-
specified in M5. Within the filamentary-ring idealization (chosen wire
cross-section, thin rings) the L/R dynamics are self-consistent -- unlike the
first-order induced-E solver in :mod:`fields.quasistatic`, this includes the
eddy loops' mutual inductance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import biot_savart, circular_loop
from spin_dynamics.fields.quasistatic import mutual_inductance, self_inductance_circular

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_gradient(segments, sample_point, axis_index, *, delta=1e-4):
    """d B_axis / d(axis) at ``sample_point`` from ``segments`` per unit current."""

    p_plus = np.asarray(sample_point, dtype=np.float64).copy()
    p_minus = p_plus.copy()
    p_plus[axis_index] += delta
    p_minus[axis_index] -= delta
    b_plus = biot_savart(p_plus[None, :], segments, 1.0)[0, axis_index]
    b_minus = biot_savart(p_minus[None, :], segments, 1.0)[0, axis_index]
    return float((b_plus - b_minus) / (2.0 * delta))


@dataclass(frozen=True)
class EddyModeSpectrum:
    """Eddy-mode decomposition of a gradient's residual field at the sample."""

    tau: np.ndarray      # (n_modes,) decay time constants (s)
    alpha: np.ndarray    # (n_modes,) fractional field residues (delivered = drive*(1 - sum alpha_k e^{-t/tau_k}))
    drive_gradient: float  # dB_axis/d(axis) of the drive coil per unit current (T/m/A)


class EddyModes:
    """Coaxial-ring eddy model of a conducting structure.

    ``radii``/``centers`` describe the rings (from
    :func:`fields.coils.cylindrical_shield`); ``wire_radius`` and ``resistivity``
    set each ring's resistance and self-inductance. Build once, then query
    :meth:`spectrum` / :meth:`to_gradient_driver` for a given drive coil.
    """

    def __init__(
        self,
        radii: Sequence[float],
        centers: np.ndarray,
        *,
        wire_radius: float,
        resistivity: float,
        axis: str = "z",
        n_segments: int = 120,
    ) -> None:
        self.radii = np.asarray(radii, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.shape != (self.radii.size, 3):
            raise ValueError("centers must have shape (n_rings, 3)")
        if axis not in _AXIS_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.axis = axis
        self._axis_index = _AXIS_INDEX[axis]
        self.wire_radius = float(wire_radius)
        self.resistivity = float(resistivity)
        self.n_segments = int(n_segments)
        self._loops = [
            circular_loop(self.centers[k], self.radii[k], axis=axis, n_segments=self.n_segments)
            for k in range(self.radii.size)
        ]
        self._build_rl()

    def _build_rl(self) -> None:
        n = self.radii.size
        # Resistance: rho * circumference / wire cross-section = rho * 2 a / r_w^2.
        self.R = np.diag(self.resistivity * 2.0 * self.radii / self.wire_radius**2)
        L = np.zeros((n, n))
        for j in range(n):
            L[j, j] = self_inductance_circular(self.radii[j], self.wire_radius)
            for k in range(j + 1, n):
                m = mutual_inductance(self._loops[j], self._loops[k])
                L[j, k] = L[k, j] = m
        self.L = L

    def spectrum(
        self,
        drive_segments: Sequence,
        *,
        sample_point: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> EddyModeSpectrum:
        """Eddy-mode (tau_k, alpha_k) of the residual gradient at ``sample_point``.

        A unit step in the drive current induces ring currents
        ``I0 = -L^{-1} M`` that decay via the modes of ``L^{-1}R``; the residual
        gradient at the sample is ``sum_k -alpha_k e^{-t/tau_k}`` relative to the
        drive gradient, so the delivered gradient is
        ``drive*(1 - sum_k alpha_k e^{-t/tau_k})``.
        """

        from scipy.linalg import eigh

        sample = np.asarray(sample_point, dtype=np.float64)
        # Drive->ring mutual inductance and per-ring field coupling.
        m_drive = np.array([mutual_inductance(loop, drive_segments) for loop in self._loops])
        c = np.array([_axis_gradient(loop, sample, self._axis_index) for loop in self._loops])
        g_drive = _axis_gradient(drive_segments, sample, self._axis_index)

        # Generalized symmetric-definite eigenproblem R v = lambda L v:
        # V^T L V = I, V^T R V = diag(lambda), lambda = 1/tau.
        lam, V = eigh(self.R, self.L)
        tau = 1.0 / lam
        # g_eddy(t) = -sum_k (V^T c)_k (V^T m_drive)_k e^{-lambda_k t}
        vc = V.T @ c
        vm = V.T @ m_drive
        beta = -(vc * vm)  # eddy gradient residue per unit drive step
        alpha = -beta / g_drive  # delivered = drive*(1 - sum alpha_k e^{-t/tau_k})
        return EddyModeSpectrum(tau=tau, alpha=alpha, drive_gradient=g_drive)

    def eddy_terms(
        self,
        drive_segments: Sequence,
        *,
        sample_point: Sequence[float] = (0.0, 0.0, 0.0),
        min_alpha: float = 1e-3,
        max_terms: int | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Return ``((alpha_k, tau_k), ...)`` for :class:`GradientDriverResponse`.

        Keeps only physically-usable modes: positive ``alpha_k`` (Lenz droop)
        above ``min_alpha``, the largest ``max_terms`` by ``alpha``. Their sum is
        clamped below 1 (the eddy field cannot exceed the drive field for a
        passive structure); a warning-free clamp keeps the driver model valid.
        """

        spec = self.spectrum(drive_segments, sample_point=sample_point)
        terms = [(a, t) for a, t in zip(spec.alpha, spec.tau) if a > min_alpha and t > 0]
        terms.sort(key=lambda at: at[0], reverse=True)
        if max_terms is not None:
            terms = terms[:max_terms]
        total = sum(a for a, _ in terms)
        if total >= 0.999:  # keep sum < 1 for GradientDriverResponse
            scale = 0.99 / total
            terms = [(a * scale, t) for a, t in terms]
        return tuple((float(a), float(t)) for a, t in terms)

    def to_gradient_driver(
        self,
        drive_segments: Sequence,
        *,
        tau_rl: float,
        sample_point: Sequence[float] = (0.0, 0.0, 0.0),
        min_alpha: float = 1e-3,
        max_terms: int | None = None,
        oversample: int = 8,
        tail=None,
    ):
        """Build a M5 ``GradientDriverResponse`` from the geometry-derived eddy terms."""

        from spin_dynamics.optimal_control.control_response import GradientDriverResponse, TailSpec

        terms = self.eddy_terms(
            drive_segments, sample_point=sample_point, min_alpha=min_alpha, max_terms=max_terms
        )
        return GradientDriverResponse(
            tau_rl=float(tau_rl),
            eddy_terms=terms,
            oversample=int(oversample),
            tail=tail if tail is not None else TailSpec(),
        )


__all__ = ["EddyModes", "EddyModeSpectrum"]
