"""Timeline visualization for native and imported pulse sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.sequences.compiler import CompiledSequence, compile_sequence
from spin_dynamics.sequences.ir import SequenceIR

TimeUnit = Literal["auto", "s", "ms", "us", "ns"]


@dataclass(frozen=True)
class SequencePlotData:
    """Scaled arrays used by :func:`plot_sequence`."""

    interval_edges: np.ndarray
    rf_i_hz: np.ndarray
    rf_q_hz: np.ndarray
    gradients_hz_per_m: np.ndarray
    adc_times: np.ndarray
    adc_phases_rad: np.ndarray
    block_boundaries: np.ndarray
    block_centers: np.ndarray
    block_labels: tuple[str, ...]
    time_unit: str
    time_scale: float
    downsampled: bool


def sequence_plot_data(
    sequence: SequenceIR | CompiledSequence,
    *,
    system_frequency_hz: float | None = None,
    time_unit: TimeUnit = "auto",
    max_points: int = 50_000,
) -> SequencePlotData:
    """Return plotting arrays without importing Matplotlib."""

    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    compiled = (
        compile_sequence(sequence, system_frequency_hz=system_frequency_hz)
        if isinstance(sequence, SequenceIR)
        else sequence
    )
    unit, scale = _resolve_time_unit(time_unit, compiled.duration_seconds)
    indices, downsampled = _plot_indices(compiled, max_points)
    if indices.size == 0:
        edges = np.asarray([0.0], dtype=np.float64)
        rf = np.empty(0, dtype=np.complex128)
        gradients = np.empty((0, 3), dtype=np.float64)
    else:
        starts = compiled.start_times_seconds[indices]
        final_end = compiled.duration_seconds
        edges = np.concatenate((starts, [final_end])) * scale
        rf = compiled.rf_hz[indices]
        gradients = compiled.gradients_hz_per_m[indices]
    block_starts = compiled.block_start_times_seconds
    block_ends = block_starts + compiled.block_durations_seconds
    boundaries = np.unique(np.concatenate((block_starts, block_ends))) * scale
    centers = (block_starts + 0.5 * compiled.block_durations_seconds) * scale
    return SequencePlotData(
        interval_edges=edges,
        rf_i_hz=np.real(rf),
        rf_q_hz=np.imag(rf),
        gradients_hz_per_m=gradients,
        adc_times=compiled.adc.times_seconds * scale,
        adc_phases_rad=compiled.adc.phase_offsets_rad.copy(),
        block_boundaries=boundaries,
        block_centers=centers,
        block_labels=compiled.block_labels,
        time_unit=unit,
        time_scale=scale,
        downsampled=downsampled,
    )


def plot_sequence(
    sequence: SequenceIR | CompiledSequence,
    *,
    system_frequency_hz: float | None = None,
    time_unit: TimeUnit = "auto",
    show_blocks: bool = True,
    max_points: int = 50_000,
    figure=None,
):
    """Plot RF I/Q, gradient channels, and ADC on one aligned time axis.

    Returns ``(figure, axes)``. Pass an existing Matplotlib figure to place the
    five lanes there; otherwise a new constrained-layout figure is created.
    """

    import matplotlib.pyplot as plt

    data = sequence_plot_data(
        sequence,
        system_frequency_hz=system_frequency_hz,
        time_unit=time_unit,
        max_points=max_points,
    )
    if figure is None:
        figure, axes = plt.subplots(
            5,
            1,
            figsize=(11.0, 7.5),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": (1.35, 1.0, 1.0, 1.0, 0.65)},
        )
    else:
        axes = figure.subplots(
            5,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": (1.35, 1.0, 1.0, 1.0, 0.65)},
        )

    _plot_step(axes[0], data.interval_edges, data.rf_i_hz, label="I")
    _plot_step(axes[0], data.interval_edges, data.rf_q_hz, label="Q")
    axes[0].set_ylabel("RF\n(Hz)")
    axes[0].legend(loc="upper right", ncols=2)

    gradient_labels = ("Gx\n(Hz/m)", "Gy\n(Hz/m)", "Gz\n(Hz/m)")
    for channel, label in enumerate(gradient_labels):
        _plot_step(
            axes[channel + 1],
            data.interval_edges,
            data.gradients_hz_per_m[:, channel],
        )
        axes[channel + 1].set_ylabel(label)

    axes[4].set_ylim(0.0, 1.0)
    axes[4].set_yticks([])
    axes[4].set_ylabel("ADC")
    if data.adc_times.size:
        axes[4].vlines(data.adc_times, 0.1, 0.9, linewidth=0.9)

    if show_blocks:
        _draw_blocks(axes, data)
    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.6, zorder=0)
        axis.grid(axis="x", color="0.85", linewidth=0.5)
        axis.margins(x=0.0)
    axes[-1].set_xlabel(f"Time ({data.time_unit})")
    if data.downsampled:
        axes[0].text(
            0.995,
            0.04,
            f"display downsampled to {max_points:,} intervals",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            color="0.35",
        )
    return figure, axes


def _plot_step(axis, edges: np.ndarray, values: np.ndarray, *, label=None) -> None:
    if values.size == 0:
        return
    axis.step(edges, np.concatenate((values, values[-1:])), where="post", label=label)


def _draw_blocks(axes, data: SequencePlotData) -> None:
    for boundary in data.block_boundaries[1:-1]:
        for axis in axes:
            axis.axvline(boundary, color="0.7", linewidth=0.65, linestyle=":")
    if len(data.block_labels) <= 24:
        for center, label in zip(data.block_centers, data.block_labels):
            axes[0].annotate(
                label,
                xy=(center, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize="x-small",
                rotation=30,
            )


def _resolve_time_unit(time_unit: TimeUnit, duration_seconds: float) -> tuple[str, float]:
    scales = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if time_unit != "auto":
        if time_unit not in scales:
            raise ValueError("time_unit must be 'auto', 's', 'ms', 'us', or 'ns'")
        return time_unit, scales[time_unit]
    if duration_seconds >= 1.0:
        return "s", 1.0
    if duration_seconds >= 1e-3:
        return "ms", 1e3
    if duration_seconds >= 1e-6:
        return "us", 1e6
    return "ns", 1e9


def _plot_indices(
    compiled: CompiledSequence, max_points: int
) -> tuple[np.ndarray, bool]:
    count = compiled.durations_seconds.size
    if count <= max_points:
        return np.arange(count, dtype=np.int64), False
    selected = np.linspace(0, count - 1, max_points, dtype=np.int64)
    return np.unique(selected), True


__all__ = ["SequencePlotData", "plot_sequence", "sequence_plot_data"]
