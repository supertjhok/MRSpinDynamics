"""One switch for the package's MATLAB-parity versus physical conventions.

Several defaults keep MATLAB/Octave parity even though a physically
consistent alternative exists (see the code review tracker, M-45, L-49 and
L-54). Each is still selectable per call; this module selects them together:

``relax_during_pulses``
    T1/T2 relaxation during RF pulses in the arb10 kernels
    (:func:`spin_dynamics.core.kernels.set_arb10_pulse_relaxation`).
``echo_registration``
    ``calc_time_domain_echo`` frequency/time registration, ``"matlab"`` or
    ``"exact"``.
``imaging_signal_convention``
    Phase-encoded imaging PAP combine scale, ``"matlab"`` (``mrx1 - mrx2``)
    or ``"cpmg"`` (``(mrx1 - mrx2) / 2``, the CPMG-train scale).
``imaging_density_normalization``
    Auxiliary off-resonance samples per voxel: ``"legacy"`` gives each the
    full voxel density, ``"preserve"`` conserves the total.

The ``"matlab"`` profile is the default and reproduces every existing
result; ``"physics"`` selects the physical alternative for all four. A
per-call argument (or, for the kernels, a ``relax_during_pulses`` params
key) always wins over the profile. The setting is process-global, like
:func:`~spin_dynamics.core.kernels.set_arb10_backend`, so it also applies
inside worker threads; the context manager is not safe for concurrent
callers that want different conventions.

PhaseCycle M+/M- detection is deliberately not part of a profile: it
describes the data being combined (the arb10 kernels acquire M-), not a
modeling choice.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Literal, Mapping

from spin_dynamics.core import kernels

EchoRegistration = Literal["matlab", "exact"]
ImagingSignalConvention = Literal["matlab", "cpmg"]
DensityNormalization = Literal["legacy", "preserve"]

_CHOICES = {
    "echo_registration": ("matlab", "exact"),
    "imaging_signal_convention": ("matlab", "cpmg"),
    "imaging_density_normalization": ("legacy", "preserve"),
}


@dataclass(frozen=True)
class Conventions:
    """A complete set of convention choices."""

    relax_during_pulses: bool = False
    echo_registration: EchoRegistration = "matlab"
    imaging_signal_convention: ImagingSignalConvention = "matlab"
    imaging_density_normalization: DensityNormalization = "legacy"

    def __post_init__(self) -> None:
        if not isinstance(self.relax_during_pulses, bool):
            raise TypeError("relax_during_pulses must be a bool")
        for name, choices in _CHOICES.items():
            if getattr(self, name) not in choices:
                options = " or ".join(repr(choice) for choice in choices)
                raise ValueError(f"{name} must be {options}")

    def as_dict(self) -> dict[str, Any]:
        """Return the choices as a JSON-ready mapping."""

        return asdict(self)


PROFILES: Mapping[str, Conventions] = {
    "matlab": Conventions(),
    "physics": Conventions(
        relax_during_pulses=True,
        echo_registration="exact",
        imaging_signal_convention="cpmg",
        imaging_density_normalization="preserve",
    ),
}

_current = PROFILES["matlab"]


def _coerce(profile: str | Conventions | Mapping[str, Any]) -> Conventions:
    if isinstance(profile, Conventions):
        return profile
    if isinstance(profile, str):
        try:
            return PROFILES[profile]
        except KeyError:
            names = ", ".join(repr(name) for name in PROFILES)
            raise ValueError(f"unknown conventions profile {profile!r}; expected {names}") from None
    return replace(get_conventions(), **dict(profile))


def get_conventions() -> Conventions:
    """Return the active conventions.

    ``relax_during_pulses`` is read from the kernels, so a direct
    :func:`~spin_dynamics.core.kernels.set_arb10_pulse_relaxation` call is
    reflected here.
    """

    return replace(_current, relax_during_pulses=kernels.get_arb10_pulse_relaxation())


def set_conventions(profile: str | Conventions | Mapping[str, Any]) -> Conventions:
    """Activate a profile name, a :class:`Conventions`, or a mapping of fields.

    A mapping changes only the fields it names. Returns the previously active
    conventions.
    """

    global _current
    previous = get_conventions()
    selected = _coerce(profile)
    kernels.set_arb10_pulse_relaxation(selected.relax_during_pulses)
    _current = selected
    return previous


@contextmanager
def conventions(profile: str | Conventions | Mapping[str, Any]) -> Iterator[Conventions]:
    """Temporarily activate ``profile`` (restored on exit).

    ``with conventions("physics"): run_ideal_cpmg_imaging(...)``.
    """

    previous = set_conventions(profile)
    try:
        yield get_conventions()
    finally:
        set_conventions(previous)


def resolve_echo_registration(value: str | None) -> str:
    """Return ``value``, or the active echo registration when ``None``."""

    return _current.echo_registration if value is None else value


def resolve_imaging_signal_convention(value: str | None) -> str:
    """Return ``value``, or the active imaging combine convention when ``None``."""

    return _current.imaging_signal_convention if value is None else value


def resolve_imaging_density_normalization(value: str | None) -> str:
    """Return ``value``, or the active density normalization when ``None``."""

    return _current.imaging_density_normalization if value is None else value


__all__ = [
    "PROFILES",
    "Conventions",
    "DensityNormalization",
    "EchoRegistration",
    "ImagingSignalConvention",
    "conventions",
    "get_conventions",
    "resolve_echo_registration",
    "resolve_imaging_density_normalization",
    "resolve_imaging_signal_convention",
    "set_conventions",
]
