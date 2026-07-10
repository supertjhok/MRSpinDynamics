"""Import and export the open, vendor-neutral Pulseq sequence format.

The initial importer supports the core event model in Pulseq 1.4.x and 1.5.x:
blocks, RF, arbitrary/default-raster gradients, trapezoids, ADC, and compressed
or uncompressed shapes.  Optional extensions are retained as block metadata;
required extensions and explicit non-default time shapes fail clearly.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Mapping

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


_DEFAULT_DEFINITIONS = {
    "AdcRasterTime": 1e-7,
    "BlockDurationRaster": 1e-5,
    "GradientRasterTime": 1e-5,
    "RadiofrequencyRasterTime": 1e-6,
}


def read_pulseq(path: str | Path) -> SequenceIR:
    """Read a Pulseq ``.seq`` text file."""

    source = Path(path)
    return parse_pulseq(source.read_text(encoding="utf-8"), source_name=str(source))


def write_pulseq(
    sequence: SequenceIR,
    path: str | Path,
    *,
    definitions: Mapping[str, Any] | None = None,
    create_signature: bool = True,
) -> None:
    """Write ``sequence`` as a core Pulseq 1.5.0 ``.seq`` text file."""

    Path(path).write_text(
        serialize_pulseq(
            sequence,
            definitions=definitions,
            create_signature=create_signature,
        ),
        encoding="utf-8",
    )


def serialize_pulseq(
    sequence: SequenceIR,
    *,
    definitions: Mapping[str, Any] | None = None,
    create_signature: bool = True,
) -> str:
    """Serialize a raster-aligned sequence as core Pulseq 1.5.0 text.

    Native IR timing is never rounded silently. Blocks, delays, and event dwells
    must align to the declared Pulseq rasters. Optional extensions and explicit
    non-default time shapes remain import-only metadata and cannot be exported.
    """

    resolved = dict(_DEFAULT_DEFINITIONS)
    resolved.update(sequence.definitions)
    if definitions is not None:
        resolved.update(definitions)
    required_extensions = resolved.pop("RequiredExtensions", None)
    if required_extensions is not None and str(required_extensions).strip():
        raise PulseqFormatError("required Pulseq extensions cannot yet be exported")
    _require_rasters(resolved, "sequence")

    block_raster = float(resolved["BlockDurationRaster"])
    rf_raster = float(resolved["RadiofrequencyRasterTime"])
    gradient_raster = float(resolved["GradientRasterTime"])
    adc_raster = float(resolved["AdcRasterTime"])
    _validate_gradient_boundaries(sequence)

    shapes: list[np.ndarray] = []
    shape_ids: dict[tuple[int, bytes], int] = {}

    def add_shape(values: np.ndarray) -> int:
        array = np.asarray(values, dtype=np.float64)
        key = (array.size, array.tobytes())
        if key not in shape_ids:
            shape_ids[key] = len(shapes) + 1
            shapes.append(array)
        return shape_ids[key]

    blocks: list[str] = []
    rf_lines: list[str] = []
    gradient_lines: list[str] = []
    adc_lines: list[str] = []
    rf_id = gradient_id = adc_id = 0

    for block_index, block in enumerate(sequence.blocks, start=1):
        if block.extensions:
            raise PulseqFormatError(
                f"block {block_index}: Pulseq extensions cannot yet be exported"
            )
        duration = _integer_units(
            block.duration_seconds, block_raster, f"block {block_index} duration"
        )

        block_rf = 0
        if block.rf is not None:
            event = block.rf
            _require_close(event.dwell_seconds, rf_raster, "RF dwell")
            rf_id += 1
            block_rf = rf_id
            amplitude = float(np.max(np.abs(event.samples_hz)))
            magnitude = np.zeros(event.samples_hz.size) if amplitude == 0.0 else np.abs(event.samples_hz) / amplitude
            phase_cycles = np.angle(event.samples_hz) / (2.0 * np.pi)
            center = event.center_seconds
            if center is None:
                weights = np.abs(event.samples_hz)
                centers = event.dwell_seconds * (
                    np.arange(event.samples_hz.size, dtype=np.float64) + 0.5
                )
                center = float(np.average(centers, weights=weights)) if np.any(weights) else 0.5 * event.duration_seconds
            rf_lines.append(
                " ".join(
                    [
                        str(rf_id),
                        _number(amplitude),
                        str(add_shape(magnitude)),
                        str(add_shape(phase_cycles)),
                        "0",
                        _number(center * 1e6),
                        str(_integer_units(event.delay_seconds, 1e-6, "RF delay")),
                        _number(event.frequency_offset_ppm),
                        _number(event.phase_offset_rad_per_mhz),
                        _number(event.frequency_offset_hz),
                        _number(event.phase_offset_rad),
                        _rf_use_code(event.use),
                    ]
                )
            )

        block_gradients: list[int] = []
        for axis, event in zip("xyz", block.gradients):
            if event is None:
                block_gradients.append(0)
                continue
            if np.isclose(event.dwell_seconds, gradient_raster, rtol=0.0, atol=1e-15):
                time_id = 0
            elif np.isclose(event.dwell_seconds, 0.5 * gradient_raster, rtol=0.0, atol=1e-15):
                time_id = -1
            else:
                raise PulseqFormatError(
                    f"G{axis} dwell must equal GradientRasterTime or half of it"
                )
            gradient_id += 1
            block_gradients.append(gradient_id)
            amplitude = float(np.max(np.abs(event.samples_hz_per_m)))
            normalized = np.zeros(event.samples_hz_per_m.size) if amplitude == 0.0 else event.samples_hz_per_m / amplitude
            gradient_lines.append(
                " ".join(
                    [
                        str(gradient_id),
                        _number(amplitude),
                        _number(event.first_hz_per_m),
                        _number(event.last_hz_per_m),
                        str(add_shape(normalized)),
                        str(time_id),
                        str(_integer_units(event.delay_seconds, 1e-6, f"G{axis} delay")),
                    ]
                )
            )

        block_adc = 0
        if block.adc is not None:
            event = block.adc
            _integer_units(event.dwell_seconds, adc_raster, "ADC dwell")
            adc_id += 1
            block_adc = adc_id
            phase_id = 0
            if event.phase_offsets_rad is not None:
                phase_id = add_shape(event.phase_offsets_rad / (2.0 * np.pi))
            adc_lines.append(
                " ".join(
                    [
                        str(adc_id),
                        str(event.num_samples),
                        _number(event.dwell_seconds * 1e9),
                        str(_integer_units(event.delay_seconds, 1e-6, "ADC delay")),
                        _number(event.frequency_offset_ppm),
                        _number(event.phase_offset_rad_per_mhz),
                        _number(event.frequency_offset_hz),
                        _number(event.phase_offset_rad),
                        str(phase_id),
                    ]
                )
            )

        blocks.append(
            " ".join(
                map(
                    str,
                    [
                        block_index,
                        duration,
                        block_rf,
                        *block_gradients,
                        block_adc,
                        0,
                    ],
                )
            )
        )

    resolved["TotalDuration"] = sequence.duration_seconds
    sections = [
        "# Pulseq sequence file generated by PythonSpinDynamics",
        "[VERSION]\nmajor 1\nminor 5\nrevision 0",
        "[DEFINITIONS]\n" + "\n".join(
            f"{key} {_definition(value)}" for key, value in sorted(resolved.items())
        ),
        "[BLOCKS]\n" + "\n".join(blocks),
    ]
    if rf_lines:
        sections.append("[RF]\n" + "\n".join(rf_lines))
    if gradient_lines:
        sections.append("[GRADIENTS]\n" + "\n".join(gradient_lines))
    if adc_lines:
        sections.append("[ADC]\n" + "\n".join(adc_lines))
    if shapes:
        shape_blocks = []
        for shape_id, values in enumerate(shapes, start=1):
            shape_blocks.append(
                f"shape_id {shape_id}\nnum_samples {values.size}\n"
                + "\n".join(_number(value) for value in values)
            )
        sections.append("[SHAPES]\n" + "\n\n".join(shape_blocks))
    text = "\n\n".join(sections) + "\n"
    if create_signature:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        text += f"\n[SIGNATURE]\nType md5\nHash {digest}\n"
    return text


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
                center,
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
            center_seconds=float(center) * 1e-6 if version[1] == 5 else None,
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
            first = last = 0.0
        elif len(fields) == 7:
            event_id, amp, first, last, shape_id, time_id, delay = fields
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
            first_hz_per_m=float(first),
            last_hz_per_m=float(last),
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


def _rf_use_code(value: str) -> str:
    return {
        "excitation": "e",
        "refocusing": "r",
        "inversion": "i",
        "saturation": "s",
        "preparation": "p",
        "other": "o",
        "undefined": "u",
    }.get(value.lower(), value[:1].lower() or "u")


def _validate_gradient_boundaries(sequence: SequenceIR) -> None:
    previous = np.zeros(3)
    for block_index, block in enumerate(sequence.blocks, start=1):
        following = np.zeros(3)
        for axis, event in enumerate(block.gradients):
            first = 0.0 if event is None else event.first_hz_per_m
            if not np.isclose(previous[axis], first, rtol=0.0, atol=1e-12):
                name = "xyz"[axis]
                raise PulseqFormatError(
                    f"block {block_index}: G{name} boundary does not connect to "
                    "the preceding block"
                )
            if event is None:
                continue
            if first != 0.0 and event.delay_seconds != 0.0:
                raise PulseqFormatError(
                    f"block {block_index}: a nonzero gradient start requires zero delay"
                )
            ends_at_block = np.isclose(
                event.end_seconds, block.duration_seconds, rtol=0.0, atol=1e-12
            )
            if event.last_hz_per_m != 0.0 and not ends_at_block:
                raise PulseqFormatError(
                    f"block {block_index}: a nonzero gradient end must align to block end"
                )
            following[axis] = event.last_hz_per_m if ends_at_block else 0.0
        previous = following
    if np.any(np.abs(previous) > 1e-12):
        raise PulseqFormatError("the sequence must end with zero gradient boundary amplitudes")


def _integer_units(value: float, quantum: float, name: str) -> int:
    units = int(round(float(value) / float(quantum)))
    tolerance = max(1e-15, abs(value) * 1e-12)
    if units < 0 or not np.isclose(
        units * quantum, value, rtol=0.0, atol=tolerance
    ):
        raise PulseqFormatError(f"{name} must align to {_number(quantum)} s")
    return units


def _require_close(value: float, expected: float, name: str) -> None:
    if not np.isclose(value, expected, rtol=0.0, atol=1e-15):
        raise PulseqFormatError(f"{name} must equal {_number(expected)} s")


def _number(value: Any) -> str:
    return format(float(value), ".17g")


def _definition(value: Any) -> str:
    if isinstance(value, str):
        return value
    if np.isscalar(value):
        return _number(value)
    return " ".join(_number(item) for item in value)


__all__ = [
    "PulseqFormatError",
    "parse_pulseq",
    "read_pulseq",
    "serialize_pulseq",
    "write_pulseq",
]
