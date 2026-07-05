"""Field-referred detector models for the signal-detection chain.

Introduces a detector-agnostic layer (magnetic field at the sensor) so that
inductive coils, SQUIDs, and OPMs differ only in a transfer shape and a
field-referred noise floor. See ``docs/non_inductive_detection.md``.

This first increment ships the abstraction plus the inductive-coil adapter that
reproduces the existing probe SNR; ``SQUIDMagnetometer`` and ``OPMMagnetometer``
follow in later increments.
"""

from .base import (
    DetectedFieldSNR,
    Detector,
    detected_field_snr,
    field_referred_from_output,
)
from .inductive import InductiveCoilDetector

__all__ = [
    "Detector",
    "DetectedFieldSNR",
    "detected_field_snr",
    "field_referred_from_output",
    "InductiveCoilDetector",
]
