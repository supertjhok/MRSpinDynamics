"""FastHenry interoperability: export a PEEC :class:`Conductor` and cross-check against it.

FastHenry (M.I.T.; maintained by FastFieldSolvers) is the de-facto reference partial-element
inductance/resistance solver. This module writes a :class:`spin_dynamics.fields.coil_peec.Conductor`
to a FastHenry ``.inp`` file and -- on Windows, where FastHenry2 is installed and registered
as a COM automation server -- runs it and returns ``L(f)`` and ``R(f)`` so the PEEC solver
can be validated directly against FastHenry.

A round wire is exported as a square bar of equal cross-sectional area (FastHenry's
conductors are rectangular); a ``cross_section="rect"`` conductor is exported exactly. Use
``cross_section="rect"`` with matching ``n_width``/``n_height`` and FastHenry ``nwinc``/
``nhinc`` for an apples-to-apples comparison.

Running requires ``pywin32`` and a registered ``FastHenry2.Document`` COM server (the
FastFieldSolvers Windows install); :func:`run_fasthenry` raises a clear error otherwise.
The ``.inp`` export (:func:`to_fasthenry_inp`) is pure text and always available.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.coil_peec import Conductor

__all__ = [
    "to_fasthenry_inp",
    "run_fasthenry",
    "FastHenryResult",
    "compare_with_fasthenry",
]


def _bar_dimensions(conductor: Conductor) -> tuple[float, float]:
    """Return the FastHenry (width, height) for the conductor's cross-section (m)."""

    if conductor.cross_section == "rect":
        return float(conductor.width), float(conductor.height)
    # Round wire -> square bar of equal cross-sectional area.
    side = float(np.sqrt(np.pi) * conductor.wire_radius)
    return side, side


def to_fasthenry_inp(
    conductor: Conductor,
    frequencies,
    *,
    nwinc: int | None = None,
    nhinc: int | None = None,
    rw: float | None = None,
    rh: float | None = None,
) -> str:
    """Return a FastHenry ``.inp`` deck for ``conductor`` over ``frequencies`` (Hz).

    Units are metres, so conductivity is written in S/m directly. The wire path becomes a
    chain of nodes/segments; the two ends are the single external port. ``nwinc``/``nhinc``
    set the cross-section filament counts (default: the conductor's own tiling counts).

    ``rw``/``rh`` set the ratio of adjacent filament sizes across the width/height. **If
    omitted, FastHenry defaults them to 2.0** (geometrically graded toward the surface --
    ``readGeom.c``'s ``DEFAULTS``), NOT uniform; pass ``rw=rh=1.0`` for an apples-to-apples
    comparison against this package's uniform (``grading=1``) tiling.
    """

    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    pts = conductor.path_points
    w, h = _bar_dimensions(conductor)
    sigma = 1.0 / conductor.material.resistivity_at(conductor.temperature)
    if conductor.cross_section == "rect":
        nw = nwinc if nwinc is not None else conductor.n_width
        nh = nhinc if nhinc is not None else conductor.n_height
    else:
        nw = nwinc if nwinc is not None else conductor.n_angular
        nh = nhinc if nhinc is not None else conductor.n_radial

    ratios = ""
    if rw is not None:
        ratios += f" rw={float(rw):.8g}"
    if rh is not None:
        ratios += f" rh={float(rh):.8g}"
    lines = [
        "* FastHenry deck generated from a spin_dynamics PEEC Conductor",
        ".units m",
        f".Default sigma={sigma:.8g} nwinc={int(nw)} nhinc={int(nh)}{ratios}",
        "",
    ]
    for i, p in enumerate(pts):
        lines.append(f"N{i} x={p[0]:.10g} y={p[1]:.10g} z={p[2]:.10g}")
    lines.append("")
    for i in range(len(pts) - 1):
        lines.append(f"E{i} N{i} N{i + 1} w={w:.10g} h={h:.10g}")
    lines.append("")
    lines.append(f".external N0 N{len(pts) - 1}")
    if freqs.size == 1:
        lines.append(f".freq fmin={freqs[0]:.8g} fmax={freqs[0]:.8g} ndec=1")
    else:
        # ndec chosen so FastHenry hits ~freqs.size points across the decade span.
        span = np.log10(freqs[-1] / freqs[0])
        ndec = max(1, round((freqs.size - 1) / span)) if span > 0 else 1
        lines.append(f".freq fmin={freqs[0]:.8g} fmax={freqs[-1]:.8g} ndec={int(ndec)}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class FastHenryResult:
    """FastHenry ``L(f)`` and ``R(f)`` for the (single-port) conductor."""

    frequency: np.ndarray   # (F,) Hz
    inductance: np.ndarray  # (F,) H
    resistance: np.ndarray  # (F,) ohm


def _find_fasthenry_com():
    """Return a dispatched FastHenry2 COM object or raise a clear error."""

    try:
        import win32com.client  # type: ignore
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "running FastHenry needs pywin32 (pip install pywin32) and a Windows "
            "FastHenry2 install; only to_fasthenry_inp() is available otherwise"
        ) from exc
    try:
        return win32com.client.Dispatch("FastHenry2.Document")
    except Exception as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "could not create the 'FastHenry2.Document' COM server; install FastHenry2 "
            "from FastFieldSolvers (which registers the automation server) to run it"
        ) from exc


def run_fasthenry(
    conductor: Conductor,
    frequencies,
    *,
    nwinc: int | None = None,
    nhinc: int | None = None,
    rw: float | None = None,
    rh: float | None = None,
    timeout: float = 300.0,
    workdir: str | None = None,
) -> FastHenryResult:
    """Run FastHenry on ``conductor`` and return its ``L(f)``/``R(f)`` (single port).

    Writes the ``.inp`` deck, drives the ``FastHenry2.Document`` COM server (dynamic
    dispatch: ``Run(path)``, poll the ``IsRunning`` property, read the ``GetInductance`` and
    ``GetResistance`` properties, ``Quit``), and returns the per-frequency inductance and
    resistance. Windows + FastHenry2 + pywin32 only; raises ``RuntimeError`` otherwise.
    """

    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    deck = to_fasthenry_inp(conductor, freqs, nwinc=nwinc, nhinc=nhinc, rw=rw, rh=rh)
    tmpdir = workdir or tempfile.mkdtemp(prefix="fasthenry_")
    inp_path = os.path.join(tmpdir, "peec_conductor.inp")
    with open(inp_path, "w", encoding="ascii") as fh:
        fh.write(deck)

    doc = _find_fasthenry_com()
    doc.Run('"%s"' % os.path.abspath(inp_path))
    t0 = time.time()
    while doc.IsRunning:
        time.sleep(0.2)
        if time.time() - t0 > timeout:  # pragma: no cover
            raise RuntimeError("FastHenry run exceeded the timeout")
    # GetInductance/GetResistance are COM properties returning (F, nport, nport) arrays.
    ind = np.array(doc.GetInductance, dtype=np.float64)
    res = np.array(doc.GetResistance, dtype=np.float64)
    try:
        doc.Quit
    except Exception:  # pragma: no cover - Quit occasionally reports via exception
        pass
    # Single port: collapse the (F, 1, 1) matrices. FastHenry returns results in ascending
    # frequency order (fmin first), matching how `.freq` is swept here.
    ind = ind.reshape(ind.shape[0], -1)[:, 0]
    res = res.reshape(res.shape[0], -1)[:, 0]
    out_freq = freqs if ind.size == freqs.size else np.asarray(freqs[: ind.size])
    return FastHenryResult(frequency=out_freq, inductance=ind, resistance=res)


def compare_with_fasthenry(
    conductor: Conductor,
    frequencies,
    *,
    nwinc: int | None = None,
    nhinc: int | None = None,
    rw: float | None = None,
    rh: float | None = None,
    formulation: str = "chain",
):
    """Return ``(peec, fasthenry)`` results for the same conductor and frequencies.

    ``peec`` is a :class:`spin_dynamics.fields.coil_peec.PEECImpedance`; ``fasthenry`` a
    :class:`FastHenryResult`. Convenience wrapper for validation scripts/tests. For an
    apples-to-apples mesh pass ``rw=rh=1.0`` (FastHenry otherwise grades its filaments with
    ratio 2 toward the surface) and ``formulation="full"`` for multi-turn geometries (the
    per-segment system FastHenry itself uses).
    """

    from spin_dynamics.fields.coil_peec import extract_impedance

    peec = extract_impedance(conductor, frequencies, formulation=formulation)
    fh = run_fasthenry(conductor, frequencies, nwinc=nwinc, nhinc=nhinc, rw=rw, rh=rh)
    return peec, fh
