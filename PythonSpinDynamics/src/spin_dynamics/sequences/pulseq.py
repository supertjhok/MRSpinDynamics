"""Import the open, vendor-neutral Pulseq text format into the sequence IR.

The initial importer supports the core event model in Pulseq 1.4.x and 1.5.x:
blocks, RF, arbitrary/default-raster gradients, trapezoids, ADC, and compressed
or uncompressed shapes.  Optional extensions are retained as block metadata;
required extensions and explicit non-default time shapes fail clearly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from spin_dynamics.sequences.ir import (
    ADCEvent,
    GradientWaveform,
    RFPulse,
    SequenceBlock,
    SequenceIR,
)


class PulseqFormatError(ValueError):
    """Raised when a Pulseq file is malformed or uses an unsupported feature."""


def read_pulseq(path: str | Path) -> SequenceIR:
    """Read a Pulseq ``.seq`` text file."""

    source = Path(path)
    return parse_pulseq(source.read_text(encoding="utf-8"), source_name=str(source))


def parse_pulseq(text: str, *, source_name: str = "<string>") -> SequenceIR:
    """Parse Pulseq 1.4/1.5 text into :class:`SequenceIR`."""

    sections = _split_sections(text)
    version = _parse_version(sections.get("VERSION", []))
    if version[0] != 1 or version[1] not in (4, 5):
        raise PulseqFormatError(
            f"{source_name}: supported Pulseq versions are 1.4.x and 1.5.x, "
            f"got {'.'.join(map(str, version))}"
        )
    definitions = _parse_definitions(sections.get("DEFINITIONS", []))
    _require_rasters(definitions, source_name)
    required = str(definitions.get("RequiredExtensions", "")).split()
    if required:
        raise PulseqFormatError(
            f"{source_name}: required Pulseq extensions are not yet supported: "
            + ", ".join(required)
        )

    shapes = _parse_shapes(sections.get("SHAPES", []), source_name)
    rf_events = _parse_rf(sections.get("RF", []), version, definitions, shapes)
    gradients = _parse_gradients(
        sections.get("GRADIENTS", []), definitions, shapes
    )
    gradients.update(_parse_trapezoids(sections.get("TRAP", []), definitions))
    adc_events = _parse_adc(sections.get("ADC", []), version, shapes)
    blocks = _parse_blocks(
        sections.get("BLOCKS", []),
        definitions,
        rf_events,
        gradients,
        adc_events,
        source_name,
    )
    return SequenceIR(
        blocks=tuple(blocks),
        definitions=definitions,
        source_format="pulseq",
        source_version=version,
    )


def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().upper()
            current = sections.setdefault(name, [])
        elif current is not None:
            current.append(line)
    return sections


def _parse_version(lines: list[str]) -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2:
            values[fields[0].lower()] = int(fields[1])
    try:
        return values["major"], values["minor"], values["revision"]
    except KeyError as exc:
        raise PulseqFormatError("Pulseq [VERSION] must define major/minor/revision") from exc


def _parse_definitions(lines: list[str]) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    for line in lines:
        if not line:
            continue
        key, separator, value = line.partition(" ")
        if not separator:
            raise PulseqFormatError(f"invalid Pulseq definition: {line!r}")
        definitions[key] = _parse_definition_value(value.strip())
    return definitions


def _parse_definition_value(value: str) -> Any:
    fields = value.split()
    try:
        numbers = [float(field) for field in fields]
    except ValueError:
        return value
    if len(numbers) == 1:
        return numbers[0]
    return tuple(numbers)


def _require_rasters(definitions: dict[str, Any], source_name: str) -> None:
    required = (
        "AdcRasterTime",
        "BlockDurationRaster",
        "GradientRasterTime",
        "RadiofrequencyRasterTime",
    )
    missing = [name for name in required if name not in definitions]
    if missing:
        raise PulseqFormatError(
            f"{source_name}: missing required Pulseq definitions: {', '.join(missing)}"
        )
    for name in required:
        if float(definitions[name]) <= 0.0:
            raise PulseqFormatError(f"{name} must be positive")


def _parse_shapes(lines: list[str], source_name: str) -> dict[int, np.ndarray]:
    shapes: dict[int, np.ndarray] = {}
    index = 0
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        header = lines[index].split()
        if len(header) != 2 or header[0].lower() != "shape_id":
            raise PulseqFormatError(f"{source_name}: expected shape_id header")
        shape_id = int(header[1])
        index += 1
        if index >= len(lines):
            raise PulseqFormatError(f"{source_name}: shape {shape_id} has no size")
        size_fields = lines[index].split()
        if len(size_fields) != 2 or size_fields[0].lower() != "num_samples":
            raise PulseqFormatError(f"{source_name}: shape {shape_id} has no num_samples")
        num_samples = int(size_fields[1])
        index += 1
        encoded: list[float] = []
        while index < len(lines):
            line = lines[index]
            if line.lower().startswith("shape_id "):
                break
            if line:
                encoded.extend(float(value) for value in line.split())
            index += 1
        shapes[shape_id] = _decompress_shape(encoded, num_samples, shape_id)
    return shapes


def _decompress_shape(encoded: list[float], num_samples: int, shape_id: int) -> np.ndarray:
    if len(encoded) == num_samples:
        return np.asarray(encoded, dtype=np.float64)
    derivative: list[float] = []
    index = 0
    while index < len(encoded):
        value = encoded[index]
        if index + 2 < len(encoded) and value == encoded[index + 1]:
            repeats = int(round(encoded[index + 2]))
            if repeats < 0 or not np.isclose(encoded[index + 2], repeats):
                raise PulseqFormatError(f"shape {shape_id} has an invalid repeat count")
            derivative.extend([value] * (repeats + 2))
            index += 3
        else:
            derivative.append(value)
            index += 1
    values = np.cumsum(np.asarray(derivative, dtype=np.float64))
    if values.size != num_samples:
        raise PulseqFormatError(
            f"shape {shape_id} expands to {values.size}, expected {num_samples} samples"
        )
    return values


def _parse_rf(
    lines: list[str],
    version: tuple[int, int, int],
    definitions: dict[str, Any],
    shapes: dict[int, np.ndarray],
) -> dict[int, RFPulse]:
    result: dict[int, RFPulse] = {}
    dwell = float(definitions["RadiofrequencyRasterTime"])
    for fields in _records(lines):
        if version[1] == 4:
            if len(fields) != 8:
                raise PulseqFormatError("Pulseq 1.4 RF records require 8 fields")
            event_id, amp, mag_id, phase_id, time_id, delay, freq, phase = fields
            freq_ppm = phase_per_mhz = 0.0
            use = "undefined"
        else:
            if len(fields) != 12:
                raise PulseqFormatError("Pulseq 1.5 RF records require 12 fields")
            (
                event_id,
                amp,
                mag_id,
                phase_id,
                time_id,
                _center,
                delay,
                freq_ppm,
                phase_per_mhz,
                freq,
                phase,
                use,
            ) = fields
        if int(time_id) != 0:
            raise PulseqFormatError("explicit Pulseq RF time shapes are not yet supported")
        magnitude = _shape(shapes, int(mag_id))
        phase_shape = _shape(shapes, int(phase_id))
        if magnitude.size != phase_shape.size:
            raise PulseqFormatError("RF magnitude and phase shapes must have equal size")
        samples = float(amp) * magnitude * np.exp(2j * np.pi * phase_shape)
        result[int(event_id)] = RFPulse(
            samples_hz=samples,
            dwell_seconds=dwell,
            delay_seconds=float(delay) * 1e-6,
            frequency_offset_hz=float(freq),
            frequency_offset_ppm=float(freq_ppm),
            phase_offset_rad=float(phase),
            phase_offset_rad_per_mhz=float(phase_per_mhz),
            use=_rf_use(str(use)),
        )
    return result


def _parse_gradients(
    lines: list[str],
    definitions: dict[str, Any],
    shapes: dict[int, np.ndarray],
) -> dict[int, GradientWaveform]:
    result: dict[int, GradientWaveform] = {}
    raster = float(definitions["GradientRasterTime"])
    for fields in _records(lines):
        if len(fields) == 5:
            event_id, amp, shape_id, time_id, delay = fields
        elif len(fields) == 7:
            event_id, amp, _first, _last, shape_id, time_id, delay = fields
        else:
            raise PulseqFormatError("Pulseq gradient records require 5 or 7 fields")
        time_id_int = int(time_id)
        if time_id_int not in (0, -1):
            raise PulseqFormatError(
                "explicit Pulseq gradient time shapes are not yet supported"
            )
        dwell = raster if time_id_int == 0 else 0.5 * raster
        result[int(event_id)] = GradientWaveform(
            samples_hz_per_m=float(amp) * _shape(shapes, int(shape_id)),
            dwell_seconds=dwell,
            delay_seconds=float(delay) * 1e-6,
        )
    return result


def _parse_trapezoids(
    lines: list[str], definitions: dict[str, Any]
) -> dict[int, GradientWaveform]:
    result: dict[int, GradientWaveform] = {}
    raster = float(definitions["GradientRasterTime"])
    for fields in _records(lines):
        if len(fields) != 6:
            raise PulseqFormatError("Pulseq trapezoid records require 6 fields")
        event_id, amp, rise_us, flat_us, fall_us, delay_us = fields
        rise = float(rise_us) * 1e-6
        flat = float(flat_us) * 1e-6
        fall = float(fall_us) * 1e-6
        total = rise + flat + fall
        count = int(round(total / raster))
        if count <= 0 or not np.isclose(count * raster, total, rtol=0.0, atol=1e-12):
            raise PulseqFormatError("trapezoid duration must align to GradientRasterTime")
        times = raster * (np.arange(count, dtype=np.float64) + 0.5)
        values = np.full(count, float(amp), dtype=np.float64)
        if rise > 0.0:
            rising = times < rise
            values[rising] = float(amp) * times[rising] / rise
        if fall > 0.0:
            falling = times >= rise + flat
            values[falling] = float(amp) * (total - times[falling]) / fall
        result[int(event_id)] = GradientWaveform(
            samples_hz_per_m=values,
            dwell_seconds=raster,
            delay_seconds=float(delay_us) * 1e-6,
        )
    return result


def _parse_adc(
    lines: list[str],
    version: tuple[int, int, int],
    shapes: dict[int, np.ndarray],
) -> dict[int, ADCEvent]:
    result: dict[int, ADCEvent] = {}
    for fields in _records(lines):
        phase_shape = None
        if version[1] == 4:
            if len(fields) != 6:
                raise PulseqFormatError("Pulseq 1.4 ADC records require 6 fields")
            event_id, num, dwell_ns, delay_us, freq, phase = fields
            freq_ppm = phase_per_mhz = 0.0
        else:
            if len(fields) != 9:
                raise PulseqFormatError("Pulseq 1.5 ADC records require 9 fields")
            (
                event_id,
                num,
                dwell_ns,
                delay_us,
                freq_ppm,
                phase_per_mhz,
                freq,
                phase,
                phase_id,
            ) = fields
            if int(phase_id) != 0:
                phase_shape = 2.0 * np.pi * _shape(shapes, int(phase_id))
        result[int(event_id)] = ADCEvent(
            num_samples=int(num),
            dwell_seconds=float(dwell_ns) * 1e-9,
            delay_seconds=float(delay_us) * 1e-6,
            frequency_offset_hz=float(freq),
            frequency_offset_ppm=float(freq_ppm),
            phase_offset_rad=float(phase),
            phase_offset_rad_per_mhz=float(phase_per_mhz),
            phase_offsets_rad=phase_shape,
        )
    return result


def _parse_blocks(
    lines: list[str],
    definitions: dict[str, Any],
    rf_events: dict[int, RFPulse],
    gradients: dict[int, GradientWaveform],
    adc_events: dict[int, ADCEvent],
    source_name: str,
) -> list[SequenceBlock]:
    result: list[SequenceBlock] = []
    raster = float(definitions["BlockDurationRaster"])
    warned_extensions = False
    for fields in _records(lines):
        if len(fields) != 8:
            raise PulseqFormatError("Pulseq block records require 8 fields")
        block_id, duration, rf_id, gx_id, gy_id, gz_id, adc_id, ext_id = map(
            int, fields
        )
        extensions: tuple[str, ...] = ()
        if ext_id != 0:
            extensions = (f"pulseq-extension-list:{ext_id}",)
            if not warned_extensions:
                warnings.warn(
                    f"{source_name}: optional Pulseq extensions are retained as metadata "
                    "but not executed",
                    UserWarning,
                    stacklevel=2,
                )
                warned_extensions = True
        try:
            result.append(
                SequenceBlock(
                    duration_seconds=duration * raster,
                    rf=_lookup(rf_events, rf_id, "RF"),
                    gradients=(
                        _lookup(gradients, gx_id, "gradient"),
                        _lookup(gradients, gy_id, "gradient"),
                        _lookup(gradients, gz_id, "gradient"),
                    ),
                    adc=_lookup(adc_events, adc_id, "ADC"),
                    label=f"pulseq_block_{block_id}",
                    extensions=extensions,
                )
            )
        except ValueError as exc:
            raise PulseqFormatError(f"{source_name}: block {block_id}: {exc}") from exc
    if not result:
        raise PulseqFormatError(f"{source_name}: Pulseq sequence has no blocks")
    return result


def _records(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines if line]


def _shape(shapes: dict[int, np.ndarray], shape_id: int) -> np.ndarray:
    try:
        return shapes[shape_id]
    except KeyError as exc:
        raise PulseqFormatError(f"unknown Pulseq shape ID {shape_id}") from exc


def _lookup(events: dict[int, Any], event_id: int, kind: str):
    if event_id == 0:
        return None
    try:
        return events[event_id]
    except KeyError as exc:
        raise PulseqFormatError(f"unknown Pulseq {kind} event ID {event_id}") from exc


def _rf_use(value: str) -> str:
    return {
        "e": "excitation",
        "r": "refocusing",
        "i": "inversion",
        "s": "saturation",
        "p": "preparation",
        "o": "other",
        "u": "undefined",
    }.get(value.lower(), value)


__all__ = ["PulseqFormatError", "parse_pulseq", "read_pulseq"]
