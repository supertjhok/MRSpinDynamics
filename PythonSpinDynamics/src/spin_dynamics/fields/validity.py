"""Validity diagnostics for quasistatic electromagnetic field models.

The field solvers in :mod:`spin_dynamics.fields` deliberately cover static and
quasistatic regimes.  This module makes the corresponding scale assumptions
quantitative.  It is a screening tool rather than an a-posteriori error
estimator: geometry, interfaces, resonances, and source topology can make a
specific problem more sensitive than any single dimensionless threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import warnings

import numpy as np

from spin_dynamics.fields.magnetostatics import MU0

EPSILON0 = 8.8541878128e-12
C0 = 299792458.0

ValidityPolicy = Literal["warn", "error", "ignore"]
ValidityRecommendation = Literal[
    "provide_context",
    "self_consistent_mqs",
    "full_wave",
]


class QuasistaticValidityWarning(UserWarning):
    """Visible warning that a quasistatic result may be outside its regime."""


class QuasistaticValidityError(RuntimeError):
    """Raised when ``validity_policy='error'`` rejects a quasistatic solve."""


@dataclass(frozen=True)
class QuasistaticThresholds:
    """Dimensionless warning thresholds used by the screening assessment.

    ``phase_span`` and ``attenuation_span`` are respectively ``beta*L`` and
    ``alpha*L`` for a homogeneous region.  A value of 0.3 corresponds to about
    17 degrees of phase or a 26 percent amplitude change across the region.
    The remaining defaults are deliberately conservative engineering screens,
    not universal accuracy bounds.
    """

    phase_span: float = 0.3
    attenuation_span: float = 0.3
    displacement_to_conduction: float = 0.1
    born_skin_ratio: float = 0.3
    self_resonance_fraction: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.phase_span,
            self.attenuation_span,
            self.displacement_to_conduction,
            self.born_skin_ratio,
            self.self_resonance_fraction,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all quasistatic thresholds must be finite and positive")


DEFAULT_QUASISTATIC_THRESHOLDS = QuasistaticThresholds()


@dataclass(frozen=True)
class QuasistaticRegion:
    """One homogeneous material span relevant to field propagation.

    ``characteristic_length_m`` should be the largest relevant propagation
    distance through this material, not automatically the entire simulation
    bounding box.  Set ``born_approximation=True`` when the existing
    first-order conductive-volume response is being used for this region.
    """

    name: str
    characteristic_length_m: float
    relative_permittivity: float = 1.0
    conductivity_s_per_m: float = 0.0
    relative_permeability: float = 1.0
    born_approximation: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("region name must not be empty")
        positive = (
            self.characteristic_length_m,
            self.relative_permittivity,
            self.relative_permeability,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("region length, permittivity, and permeability must be positive")
        if (
            not np.isfinite(self.conductivity_s_per_m)
            or self.conductivity_s_per_m < 0.0
        ):
            raise ValueError("region conductivity must be finite and non-negative")


@dataclass(frozen=True)
class QuasistaticRegionMetrics:
    """Propagation and conductive-response scales for one material region."""

    region: QuasistaticRegion
    attenuation_per_m: float
    phase_constant_per_m: float
    attenuation_span: float
    phase_span: float
    displacement_to_conduction_ratio: float
    skin_depth_m: float
    born_skin_ratio: float


@dataclass(frozen=True)
class QuasistaticFinding:
    """One actionable reason a quasistatic result needs caution."""

    criterion: str
    message: str
    recommendation: ValidityRecommendation
    value: float | None = None
    threshold: float | None = None
    region: str | None = None


@dataclass(frozen=True)
class QuasistaticAssessment:
    """Structured validity result for a quasistatic field calculation."""

    frequency_hz: float | None
    coil_extent_m: float | None
    coil_phase_span: float | None
    self_resonant_frequency_hz: float | None
    self_resonance_fraction: float | None
    region_metrics: tuple[QuasistaticRegionMetrics, ...]
    findings: tuple[QuasistaticFinding, ...]
    thresholds: QuasistaticThresholds

    @property
    def ok(self) -> bool:
        """Whether no screening finding was raised."""

        return not self.findings

    @property
    def assessed(self) -> bool:
        """Whether frequency and at least one length scale were available."""

        return self.frequency_hz is not None and (
            self.coil_extent_m is not None or bool(self.region_metrics)
        )

    @property
    def requires_full_wave(self) -> bool:
        """Whether any finding recommends a full-wave Maxwell solve."""

        return any(f.recommendation == "full_wave" for f in self.findings)

    @property
    def requires_self_consistent_mqs(self) -> bool:
        """Whether conductive reaction fields exceed the first-order screen."""

        return any(
            f.recommendation == "self_consistent_mqs" for f in self.findings
        )

    def summary(self, solver_name: str = "field solver") -> str:
        """Return one concise user-facing summary of all findings."""

        if self.ok:
            return f"{solver_name}: quasistatic screening checks passed"
        details = "; ".join(finding.message for finding in self.findings)
        return f"{solver_name} uses a quasistatic field model: {details}"


def _region_metrics(
    region: QuasistaticRegion,
    frequency_hz: float,
) -> QuasistaticRegionMetrics:
    omega = 2.0 * np.pi * frequency_hz
    epsilon = EPSILON0 * region.relative_permittivity
    mu = MU0 * region.relative_permeability
    sigma = region.conductivity_s_per_m
    # For exp(+i*omega*t), gamma = alpha + i*beta in a passive medium.
    gamma = np.sqrt(1.0j * omega * mu * (sigma + 1.0j * omega * epsilon))
    alpha = float(max(np.real(gamma), 0.0))
    beta = float(abs(np.imag(gamma)))
    length = region.characteristic_length_m
    if omega == 0.0:
        displacement_ratio = 0.0
        skin_depth = float("inf")
        skin_ratio = 0.0
    elif sigma == 0.0:
        displacement_ratio = float("inf")
        skin_depth = float("inf")
        skin_ratio = 0.0
    else:
        displacement_ratio = float(omega * epsilon / sigma)
        skin_depth = float(np.sqrt(2.0 / (omega * mu * sigma)))
        skin_ratio = float(length / skin_depth)
    return QuasistaticRegionMetrics(
        region=region,
        attenuation_per_m=alpha,
        phase_constant_per_m=beta,
        attenuation_span=alpha * length,
        phase_span=beta * length,
        displacement_to_conduction_ratio=displacement_ratio,
        skin_depth_m=skin_depth,
        born_skin_ratio=skin_ratio,
    )


def assess_quasistatic_validity(
    frequency_hz: float | None,
    *,
    coil_extent_m: float | None = None,
    regions: Sequence[QuasistaticRegion] | None = None,
    self_resonant_frequency_hz: float | None = None,
    thresholds: QuasistaticThresholds = DEFAULT_QUASISTATIC_THRESHOLDS,
) -> QuasistaticAssessment:
    """Screen wavelength, attenuation, MQS, Born, and resonance assumptions.

    ``regions=None`` means that material context was not supplied and produces
    an explicit finding.  Pass an empty sequence to state that the calculation
    is intentionally assessed as an unloaded air/vacuum problem.
    """

    if frequency_hz is None:
        frequency = None
    else:
        frequency = float(frequency_hz)
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
    if coil_extent_m is not None:
        coil_extent = float(coil_extent_m)
        if not np.isfinite(coil_extent) or coil_extent <= 0.0:
            raise ValueError("coil_extent_m must be finite and positive")
    else:
        coil_extent = None
    if self_resonant_frequency_hz is not None:
        resonance = float(self_resonant_frequency_hz)
        if not np.isfinite(resonance) or resonance <= 0.0:
            raise ValueError("self_resonant_frequency_hz must be finite and positive")
    else:
        resonance = None

    findings: list[QuasistaticFinding] = []
    metrics: tuple[QuasistaticRegionMetrics, ...] = ()
    coil_phase: float | None = None
    resonance_fraction: float | None = None
    if frequency is None:
        findings.append(
            QuasistaticFinding(
                criterion="frequency_context",
                message=(
                    "validity was not assessed because no RF frequency was supplied; "
                    "provide frequency and material-region context or explicitly ignore "
                    "the check"
                ),
                recommendation="provide_context",
            )
        )
    else:
        if coil_extent is not None:
            coil_phase = 2.0 * np.pi * frequency * coil_extent / C0
            if coil_phase >= thresholds.phase_span:
                findings.append(
                    QuasistaticFinding(
                        criterion="coil_electrical_size",
                        message=(
                            f"free-space phase span across the coil is {coil_phase:.3g} rad "
                            f"(screen {thresholds.phase_span:.3g})"
                        ),
                        recommendation="full_wave",
                        value=coil_phase,
                        threshold=thresholds.phase_span,
                    )
                )
        if regions is None:
            findings.append(
                QuasistaticFinding(
                    criterion="material_context",
                    message=(
                        "sample/material context was not supplied, so dielectric wavelength "
                        "and loading effects remain unassessed"
                    ),
                    recommendation="provide_context",
                )
            )
        else:
            metrics = tuple(_region_metrics(region, frequency) for region in regions)
            for result in metrics:
                region = result.region
                displacement_dominant = (
                    result.displacement_to_conduction_ratio
                    >= thresholds.displacement_to_conduction
                )
                if result.phase_span >= thresholds.phase_span:
                    recommendation: ValidityRecommendation = (
                        "full_wave"
                        if displacement_dominant
                        else "self_consistent_mqs"
                    )
                    findings.append(
                        QuasistaticFinding(
                            criterion="material_phase_span",
                            message=(
                                f"{region.name!r} phase span is {result.phase_span:.3g} rad "
                                f"(screen {thresholds.phase_span:.3g})"
                            ),
                            recommendation=recommendation,
                            value=result.phase_span,
                            threshold=thresholds.phase_span,
                            region=region.name,
                        )
                    )
                if result.attenuation_span >= thresholds.attenuation_span:
                    recommendation = (
                        "full_wave"
                        if displacement_dominant
                        else "self_consistent_mqs"
                    )
                    findings.append(
                        QuasistaticFinding(
                            criterion="material_attenuation_span",
                            message=(
                                f"{region.name!r} attenuation span is "
                                f"{result.attenuation_span:.3g} nepers "
                                f"(screen {thresholds.attenuation_span:.3g})"
                            ),
                            recommendation=recommendation,
                            value=result.attenuation_span,
                            threshold=thresholds.attenuation_span,
                            region=region.name,
                        )
                    )
                if region.born_approximation:
                    if displacement_dominant:
                        findings.append(
                            QuasistaticFinding(
                                criterion="mqs_displacement_current",
                                message=(
                                    f"{region.name!r} has omega*epsilon/sigma="
                                    f"{result.displacement_to_conduction_ratio:.3g} "
                                    f"(MQS screen {thresholds.displacement_to_conduction:.3g})"
                                ),
                                recommendation="full_wave",
                                value=result.displacement_to_conduction_ratio,
                                threshold=thresholds.displacement_to_conduction,
                                region=region.name,
                            )
                        )
                    if result.born_skin_ratio >= thresholds.born_skin_ratio:
                        findings.append(
                            QuasistaticFinding(
                                criterion="born_reaction_field",
                                message=(
                                    f"{region.name!r} has L/skin_depth="
                                    f"{result.born_skin_ratio:.3g} "
                                    f"(Born screen {thresholds.born_skin_ratio:.3g})"
                                ),
                                recommendation=(
                                    "full_wave"
                                    if displacement_dominant
                                    else "self_consistent_mqs"
                                ),
                                value=result.born_skin_ratio,
                                threshold=thresholds.born_skin_ratio,
                                region=region.name,
                            )
                        )
        if resonance is not None:
            resonance_fraction = frequency / resonance
            if resonance_fraction >= thresholds.self_resonance_fraction:
                findings.append(
                    QuasistaticFinding(
                        criterion="distributed_resonance",
                        message=(
                            f"operating frequency is {resonance_fraction:.3g} of the estimated "
                            f"self-resonance (screen "
                            f"{thresholds.self_resonance_fraction:.3g})"
                        ),
                        recommendation="full_wave",
                        value=resonance_fraction,
                        threshold=thresholds.self_resonance_fraction,
                    )
                )

    return QuasistaticAssessment(
        frequency_hz=frequency,
        coil_extent_m=coil_extent,
        coil_phase_span=coil_phase,
        self_resonant_frequency_hz=resonance,
        self_resonance_fraction=resonance_fraction,
        region_metrics=metrics,
        findings=tuple(findings),
        thresholds=thresholds,
    )


def apply_validity_policy(
    assessment: QuasistaticAssessment,
    *,
    solver_name: str,
    policy: ValidityPolicy = "warn",
    stacklevel: int = 2,
) -> None:
    """Warn, raise, or deliberately ignore actionable assessment findings."""

    if policy not in {"warn", "error", "ignore"}:
        raise ValueError("validity policy must be 'warn', 'error', or 'ignore'")
    if policy == "ignore" or assessment.ok:
        return
    message = assessment.summary(solver_name)
    if policy == "error":
        raise QuasistaticValidityError(message)
    warnings.warn(
        message,
        QuasistaticValidityWarning,
        stacklevel=stacklevel,
    )


__all__ = [
    "C0",
    "EPSILON0",
    "DEFAULT_QUASISTATIC_THRESHOLDS",
    "QuasistaticAssessment",
    "QuasistaticFinding",
    "QuasistaticRegion",
    "QuasistaticRegionMetrics",
    "QuasistaticThresholds",
    "QuasistaticValidityError",
    "QuasistaticValidityWarning",
    "ValidityPolicy",
    "apply_validity_policy",
    "assess_quasistatic_validity",
]
