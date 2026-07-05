"""Field-referred detector models for the signal-detection chain.

Introduces a detector-agnostic layer (magnetic field at the sensor) so that
inductive coils, SQUIDs, and OPMs differ only in a transfer shape and a
field-referred noise floor. See ``docs/non_inductive_detection.md``.

Ships the field-referred abstraction, inductive-coil detectors (a measured-probe
adapter and an idealized ``1/f`` Faraday baseline), and the non-inductive
``SQUIDMagnetometer`` and ``OPMMagnetometer``.
"""

from .base import (
    DetectedFieldSNR,
    Detector,
    detected_field_snr,
    field_referred_from_output,
)
from .gradiometer import Gradiometer
from .inductive import IdealFaradayCoil, InductiveCoilDetector
from .opm import OPMMagnetometer
from .squid import SQUIDMagnetometer

__all__ = [
    "Detector",
    "DetectedFieldSNR",
    "detected_field_snr",
    "field_referred_from_output",
    "InductiveCoilDetector",
    "IdealFaradayCoil",
    "SQUIDMagnetometer",
    "OPMMagnetometer",
    "Gradiometer",
]
