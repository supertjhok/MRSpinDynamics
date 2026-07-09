"""FasterCap interoperability: panelize a PEEC :class:`Conductor` and cross-check capacitance.

FasterCap (FastFieldSolvers, the successor to M.I.T. FastCap) is the reference boundary-element
capacitance solver. This module panelizes a conductor's surface into the FasterCap panel
format and -- on Windows, where FasterCap is installed and its COM automation server is
registered -- runs it and returns the self-capacitance, so the PEEC electrostatic kernel
(:func:`spin_dynamics.fields.coil_peec.capacitance_to_ground`) can be validated directly.

The wire surface is swept along the conductor path (rotation-minimizing frames) as a tube of
quadrilateral panels with triangular end caps -- exact for a straight round wire, and usable
for open coils. Round cross-section only for now.

Running requires ``pywin32`` and a registered ``FasterCap.Document`` COM server. If the
FasterCap type library is not registered (so method names do not resolve), register it once
from an elevated 32-bit PowerShell via ``LoadTypeLibEx`` on ``IFasterCap.tlb``; see
``docs/coil_peec.md``. The panel export (:func:`to_fastercap_panels`) is pure text and
always available.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from spin_dynamics.fields.coil_peec import Conductor, _rotation_minimizing_frames

__all__ = [
    "to_fastercap_panels",
    "run_fastercap",
    "compare_capacitance_with_fastercap",
]


def to_fastercap_panels(
    conductor: Conductor, *, n_theta: int = 24, name: str = "cond"
) -> str:
    """Return a FasterCap panel deck for the conductor's (round) wire surface.

    The surface is a tube of ``Q`` (quad) side panels swept along the path with ``n_theta``
    facets around, closed by ``T`` (triangle) fans at the two ends. Coordinates are in
    metres (so the returned capacitance is in farads). Panel vertices are ordered for an
    outward normal.
    """

    if conductor.cross_section != "round":
        raise NotImplementedError("FasterCap export currently supports round wires only")
    pts = conductor.path_points
    a = conductor.wire_radius
    e1, e2 = _rotation_minimizing_frames(pts)
    th = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    cos, sin = np.cos(th), np.sin(th)
    # Ring of surface points at each path vertex: (M+1, n_theta, 3)
    rings = pts[:, None, :] + a * (cos[None, :, None] * e1[:, None, :] + sin[None, :, None] * e2[:, None, :])

    def fmt(*vecs):
        return " ".join("%.9g" % v for vec in vecs for v in vec)

    lines = ["0 PEEC conductor surface for FasterCap"]
    m = pts.shape[0] - 1
    for k in range(m):
        for j in range(int(n_theta)):
            jn = (j + 1) % int(n_theta)
            lines.append(f"Q {name} {fmt(rings[k, j], rings[k, jn], rings[k + 1, jn], rings[k + 1, j])}")
    # End caps: fan from each end vertex (outward normals: -tangent at start, +tangent at end).
    for j in range(int(n_theta)):
        jn = (j + 1) % int(n_theta)
        lines.append(f"T {name} {fmt(pts[0], rings[0, jn], rings[0, j])}")
        lines.append(f"T {name} {fmt(pts[-1], rings[-1, j], rings[-1, jn])}")
    return "\n".join(lines) + "\n"


def _find_fastercap_com():
    """Return an early-bound FasterCap COM object or raise a clear error."""

    try:
        import win32com.client  # type: ignore
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "running FasterCap needs pywin32 (pip install pywin32) and a Windows FasterCap "
            "install; only to_fastercap_panels() is available otherwise"
        ) from exc
    try:
        # Early binding needs the type library registered; falls back to late binding.
        try:
            return win32com.client.gencache.EnsureDispatch("FasterCap.Document")
        except Exception:
            return win32com.client.Dispatch("FasterCap.Document")
    except Exception as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "could not create the 'FasterCap.Document' COM server; install FasterCap and, "
            "if method names do not resolve, register IFasterCap.tlb (see docs/coil_peec.md)"
        ) from exc


def run_fastercap(
    conductor: Conductor,
    *,
    n_theta: int = 24,
    tolerance: float = 0.01,
    timeout: float = 180.0,
    workdir: str | None = None,
) -> float:
    """Panelize ``conductor``, run FasterCap and return its self-capacitance (F).

    ``tolerance`` is FasterCap's automatic mesh-refinement tolerance (``-a``). Windows +
    FasterCap + pywin32 with a registered type library only; raises ``RuntimeError``
    otherwise. See the module docstring for the type-library registration step.
    """

    deck = to_fastercap_panels(conductor, n_theta=n_theta)
    tmpdir = workdir or tempfile.mkdtemp(prefix="fastercap_")
    path = os.path.join(tmpdir, "peec_conductor.txt")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(deck)

    doc = _find_fastercap_com()
    doc.Run('"%s" -a%g' % (os.path.abspath(path), tolerance))
    t0 = time.time()
    while doc.IsRunning():
        time.sleep(0.2)
        if time.time() - t0 > timeout:  # pragma: no cover
            raise RuntimeError("FasterCap run exceeded the timeout")
    cap = np.array(doc.GetCapacitance(), dtype=np.float64)
    try:
        doc.Quit()
    except Exception:  # pragma: no cover
        pass
    return float(cap.reshape(-1)[0])


def compare_capacitance_with_fastercap(
    conductor: Conductor, *, n_theta: int = 24, tolerance: float = 0.01
) -> tuple[float, float]:
    """Return ``(peec, fastercap)`` self-capacitance (F) for the same conductor.

    ``peec`` is :func:`spin_dynamics.fields.coil_peec.capacitance_to_ground`; ``fastercap``
    is the boundary-element result. Convenience wrapper for validation scripts/tests.
    """

    from spin_dynamics.fields.coil_peec import capacitance_to_ground

    return capacitance_to_ground(conductor), run_fastercap(conductor, n_theta=n_theta, tolerance=tolerance)
