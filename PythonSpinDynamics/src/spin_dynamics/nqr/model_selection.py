"""Choose between the reduced two-level and full density-matrix NQR models.

This implements the model-selection procedure from the technical note
(``References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite``). The reduced
fictitious-spin-1/2 path (``SelectivePulse``, ``simulate_slse``) is honest only
when the RF pulse addresses a *single isolated pair of non-degenerate states*.
The decision is read from the actual static Hamiltonian and the RF matrix
elements for the coil polarization -- not from the spin or the number of
spectral lines:

* the pulse-addressed connected set (states reachable from the target pair via
  RF-active transitions the pulse can drive, plus unresolved degenerate
  partners) must contain exactly the two target states, and
* the spacing to the nearest competing RF-active transition must dominate the
  pulse's broadening scale, ``Delta_iso >> max(Omega_1, 1/t_p, Gamma)``.

A Kramers doublet (spin-3/2 and higher half-integer spins at zero field) makes
the target endpoints degenerate, so the addressed set spans four or more states
and the full model is required. For spin-1, ``eta = 0`` collapses two
transitions onto one frequency (``Delta_iso = 0``); a small nonzero ``eta`` may
still leave the lines unresolved, which the isolation ratio -- not the logical
test ``eta != 0`` -- detects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.systems import NQRTransition, QuadrupolarSite


@dataclass(frozen=True)
class NQRModelSelection:
    """Recommendation and diagnostics for the reduced-vs-full modeling choice."""

    recommended_model: str  # "reduced" or "full"
    reduced_is_valid: bool
    target_label: str
    target_frequency_hz: float
    target_states: tuple[int, int]
    target_coupling: float
    target_is_rf_dark: bool
    carrier_frequency_hz: float
    nearest_competing_label: str | None
    nearest_competing_frequency_hz: float | None
    isolation_hz: float
    effective_nutation_hz: float
    pulse_bandwidth_hz: float
    linewidth_hz: float
    broadening_hz: float
    isolation_ratio: float
    isolation_threshold: float
    active_states: tuple[int, ...]
    degenerate_target: bool
    reasons: tuple[str, ...]

    def describe(self) -> str:
        """Return a human-readable diagnostic report."""

        def hz(value: float | None) -> str:
            if value is None:
                return "n/a"
            if not np.isfinite(value):
                return "inf"
            if abs(value) >= 1e6:
                return f"{value / 1e6:.4g} MHz"
            if abs(value) >= 1e3:
                return f"{value / 1e3:.4g} kHz"
            return f"{value:.4g} Hz"

        competitor = (
            f"{self.nearest_competing_label} @ "
            f"{hz(self.nearest_competing_frequency_hz)}"
            if self.nearest_competing_label is not None
            else "none (no RF-active competitor)"
        )
        lines = [
            f"recommended model : {self.recommended_model.upper()}",
            f"target transition : {self.target_label} @ "
            f"{hz(self.target_frequency_hz)} (levels {self.target_states})",
            f"target coupling   : {self.target_coupling:.4g}"
            + ("  [RF-dark for this polarization]" if self.target_is_rf_dark else ""),
            f"carrier           : {hz(self.carrier_frequency_hz)}",
            f"nearest competitor: {competitor}",
            f"isolation Delta   : {hz(self.isolation_hz)}",
            f"broadening max of : Omega_1={hz(self.effective_nutation_hz)}, "
            f"1/t_p={hz(self.pulse_bandwidth_hz)}, Gamma={hz(self.linewidth_hz)} "
            f"-> {hz(self.broadening_hz)}",
            f"isolation ratio   : {self.isolation_ratio:.3g} "
            f"(threshold {self.isolation_threshold:.3g})",
            f"addressed states  : {self.active_states}"
            + ("  [degenerate target]" if self.degenerate_target else ""),
            "reasons:",
        ]
        lines.extend(f"  - {reason}" for reason in self.reasons)
        return "\n".join(lines)


def _unit(direction) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("b1_direction_pas must be a finite non-zero vector")
    return vec / norm


def select_nqr_model(
    site: QuadrupolarSite,
    target: str | NQRTransition,
    *,
    nutation_hz: float,
    pulse_duration_seconds: float,
    b1_direction_pas=(1.0, 0.0, 0.0),
    linewidth_hz: float = 0.0,
    b0_vector_tesla_pas=None,
    rf_frequency_hz: float | None = None,
    isolation_threshold: float = 5.0,
    coupling_tolerance: float = 1e-2,
) -> NQRModelSelection:
    """Recommend the reduced or full NQR model for a pulse on one transition.

    ``nutation_hz`` is the bare field nutation ``gamma * B1 / (2 pi)`` (the same
    convention as ``spin_dynamics.nqr.full_dynamics``); the target's effective
    nutation rate is ``2 * nutation_hz * |<a| e1 . I |b>|``. ``linewidth_hz`` is
    the sample linewidth/inhomogeneous spread ``Gamma``. The reduced model is
    recommended only when the target is an isolated, non-degenerate, RF-active
    pair and ``isolation_ratio >= isolation_threshold``.
    """

    if nutation_hz < 0 or not np.isfinite(nutation_hz):
        raise ValueError("nutation_hz must be non-negative and finite")
    if linewidth_hz < 0 or not np.isfinite(linewidth_hz):
        raise ValueError("linewidth_hz must be non-negative and finite")
    if isolation_threshold <= 0:
        raise ValueError("isolation_threshold must be positive")

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    if isinstance(target, NQRTransition):
        target_transition = target
    else:
        target_transition = eigensystem.transition(target)
    b1 = _unit(b1_direction_pas)
    levels = eigensystem.levels_hz

    def coupling(transition: NQRTransition) -> float:
        return float(abs(np.vdot(b1, transition.dipole_vector)))

    target_coupling = coupling(target_transition)
    reference_coupling = max(
        (coupling(t) for t in eigensystem.transitions), default=0.0
    )
    active_floor = coupling_tolerance * reference_coupling
    target_is_rf_dark = reference_coupling <= 0 or target_coupling <= active_floor

    carrier = (
        float(target_transition.frequency_hz)
        if rf_frequency_hz is None
        else float(rf_frequency_hz)
    )
    effective_nutation = 2.0 * float(nutation_hz) * target_coupling
    if pulse_duration_seconds > 0:
        pulse_bandwidth = 1.0 / float(pulse_duration_seconds)
    else:
        pulse_bandwidth = np.inf
    broadening = max(effective_nutation, pulse_bandwidth, float(linewidth_hz))
    if broadening <= 0:
        broadening = 0.0

    competitors = [
        t
        for t in eigensystem.transitions
        if t is not target_transition and coupling(t) > active_floor
    ]
    if competitors:
        nearest = min(competitors, key=lambda t: abs(t.frequency_hz - carrier))
        nearest_label: str | None = nearest.label
        nearest_frequency: float | None = float(nearest.frequency_hz)
        isolation_hz = float(abs(nearest.frequency_hz - carrier))
    else:
        nearest_label = None
        nearest_frequency = None
        isolation_hz = np.inf
    isolation_ratio = (
        isolation_hz / broadening if broadening > 0 else np.inf
    )

    # A level is "RF-driven" if the pulse drives at least one RF-active
    # transition touching it (within the broadening of the carrier). A
    # near-degenerate partner only enters the addressed manifold if it is itself
    # driven -- this keeps a Kramers doublet (both members driven at the same
    # frequency) as a four-state system, while a partner reached only through an
    # RF-dark transition stays a spectator (polarization-selective case).
    driven = [
        any(
            coupling(t) > active_floor
            and abs(t.frequency_hz - carrier) < broadening
            and level in (t.lower, t.upper)
            for t in eigensystem.transitions
        )
        for level in range(levels.size)
    ]

    # Pulse-addressed connected set: states reachable from the target pair via
    # driven RF-active transitions, plus unresolved *driven* degenerate partners.
    active: set[int] = {target_transition.lower, target_transition.upper}
    changed = True
    while changed:
        changed = False
        for transition in eigensystem.transitions:
            if coupling(transition) <= active_floor:
                continue
            if abs(transition.frequency_hz - carrier) >= broadening:
                continue
            if transition.lower in active or transition.upper in active:
                for state in (transition.lower, transition.upper):
                    if state not in active:
                        active.add(state)
                        changed = True
        for level in range(levels.size):
            if level in active or not driven[level]:
                continue
            if any(abs(levels[level] - levels[m]) < broadening for m in active):
                active.add(level)
                changed = True
    active_states = tuple(sorted(active))

    endpoints = (target_transition.lower, target_transition.upper)
    degenerate_target = any(
        level not in endpoints
        and driven[level]
        and (
            abs(levels[level] - levels[endpoints[0]]) < broadening
            or abs(levels[level] - levels[endpoints[1]]) < broadening
        )
        for level in range(levels.size)
    )

    single_pair = set(active_states) == set(endpoints)
    reduced_is_valid = (
        single_pair
        and not target_is_rf_dark
        and isolation_ratio >= isolation_threshold
    )

    reasons: list[str] = []
    if target_is_rf_dark:
        reasons.append(
            "target transition is RF-dark for this B1 polarization; the pulse "
            "cannot drive it (check the coil orientation or target line)"
        )
    if degenerate_target:
        half_integer = abs(round(2 * site.spin) % 2) == 1
        kind = "Kramers doublet" if half_integer else "near-degenerate levels"
        reasons.append(
            f"target shares an energy within the pulse broadening of another "
            f"level ({kind}); the addressed manifold exceeds two states"
        )
    if not single_pair and not degenerate_target:
        reasons.append(
            f"pulse-addressed connected set spans {len(active_states)} states "
            f"{active_states}; more than one transition is driven"
        )
    if np.isfinite(isolation_ratio) and isolation_ratio < isolation_threshold:
        reasons.append(
            f"isolation ratio {isolation_ratio:.2g} < threshold "
            f"{isolation_threshold:.2g}: competing RF-active transition "
            f"'{nearest_label}' lies within reach of the carrier"
        )
    if reduced_is_valid:
        reasons.append(
            f"target is an isolated, non-degenerate, RF-active pair "
            f"(isolation ratio {isolation_ratio:.3g}); the reduced two-level "
            f"model is valid"
        )
    elif not reasons:
        reasons.append("reduced-model conditions not met; use the full model")

    return NQRModelSelection(
        recommended_model="reduced" if reduced_is_valid else "full",
        reduced_is_valid=reduced_is_valid,
        target_label=target_transition.label,
        target_frequency_hz=float(target_transition.frequency_hz),
        target_states=endpoints,
        target_coupling=target_coupling,
        target_is_rf_dark=target_is_rf_dark,
        carrier_frequency_hz=carrier,
        nearest_competing_label=nearest_label,
        nearest_competing_frequency_hz=nearest_frequency,
        isolation_hz=isolation_hz,
        effective_nutation_hz=effective_nutation,
        pulse_bandwidth_hz=pulse_bandwidth,
        linewidth_hz=float(linewidth_hz),
        broadening_hz=broadening,
        isolation_ratio=isolation_ratio,
        isolation_threshold=float(isolation_threshold),
        active_states=active_states,
        degenerate_target=degenerate_target,
        reasons=tuple(reasons),
    )
