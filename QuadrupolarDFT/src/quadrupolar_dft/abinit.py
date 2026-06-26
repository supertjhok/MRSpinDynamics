"""ABINIT EFG input and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .tensors import EFGTensor


@dataclass(frozen=True)
class AbinitEFGRecord:
    """EFG result for one ABINIT atom block."""

    atom_index: int
    typat: int
    quadrupole_moment_barns: float
    cq_mhz: float
    eta: float
    eigvals_au: np.ndarray
    eigvals_1e21_v_per_m2: np.ndarray
    eigvecs: np.ndarray
    total_tensor_au: np.ndarray | None = None

    @property
    def tensor(self) -> EFGTensor | None:
        if self.total_tensor_au is None:
            return None
        return EFGTensor.from_components(
            self.total_tensor_au,
            unit="au",
            trace_tolerance=1e-5,
        )


def parse_abinit_efg(text: str) -> list[AbinitEFGRecord]:
    """Parse ABINIT ``Electric Field Gradient Calculation`` output blocks."""

    records: list[AbinitEFGRecord] = []
    atom_matches = list(_ATOM_RE.finditer(text))
    for index, match in enumerate(atom_matches):
        start = match.start()
        if index + 1 < len(atom_matches):
            end = atom_matches[index + 1].start()
        else:
            end = len(text)
        block = text[start:end]
        atom_index = _match_int(match, "atom")
        typat = _match_int(match, "typat")
        nuclear = _NUCLEAR_RE.search(block)
        if nuclear is not None:
            quadrupole_moment_barns = float(nuclear.group("q"))
            cq_mhz = float(nuclear.group("cq"))
            eta = float(nuclear.group("eta"))
        elif match.group("cq") is not None:
            quadrupole_moment_barns = float("nan")
            cq_mhz = float(match.group("cq"))
            eta = float(match.group("eta"))
        else:
            continue

        eigvals_au = []
        eigvals_si = []
        eigvecs = []
        for eig_match in _EIG_RE.finditer(block):
            eigvals_au.append(float(eig_match.group("au")))
            si_value = eig_match.group("si")
            eigvals_si.append(float(si_value) if si_value is not None else float("nan"))
            eigvecs.append(
                [
                    float(eig_match.group("x")),
                    float(eig_match.group("y")),
                    float(eig_match.group("z")),
                ]
            )

        total_tensor = _parse_named_tensor(block, "total efg")
        records.append(
            AbinitEFGRecord(
                atom_index=atom_index,
                typat=typat,
                quadrupole_moment_barns=quadrupole_moment_barns,
                cq_mhz=cq_mhz,
                eta=eta,
                eigvals_au=np.asarray(eigvals_au, dtype=float),
                eigvals_1e21_v_per_m2=np.asarray(eigvals_si, dtype=float),
                eigvecs=np.asarray(eigvecs, dtype=float),
                total_tensor_au=total_tensor,
            )
        )
    return records


def format_abinit_efg_block(
    quadrupole_moments_barns: Sequence[float],
    *,
    nucefg: int = 2,
    point_charges: Sequence[float] | None = None,
) -> str:
    """Return the ABINIT input lines that request an EFG calculation."""

    if nucefg not in {1, 2, 3}:
        raise ValueError("ABINIT nucefg must be 1, 2, or 3.")
    lines = [
        "# Electric-field-gradient and quadrupolar-coupling output",
        "# Requires PAW datasets; norm-conserving pseudopotentials are unsuitable.",
        f"nucefg {nucefg}",
        "quadmom " + _format_values(quadrupole_moments_barns),
    ]
    if point_charges is not None:
        if nucefg != 3:
            raise ValueError("point_charges require nucefg=3.")
        lines.append("ptcharge " + _format_values(point_charges))
    return "\n".join(lines) + "\n"


def _parse_named_tensor(block: str, name: str) -> np.ndarray | None:
    rows: list[list[float]] = []
    row_re = re.compile(
        rf"^\s*{re.escape(name)}\s*:\s+"
        rf"(?P<a>{_FLOAT})\s+(?P<b>{_FLOAT})\s+(?P<c>{_FLOAT})\s*$",
        re.MULTILINE,
    )
    for row_match in row_re.finditer(block):
        rows.append(
            [
                float(row_match.group("a")),
                float(row_match.group("b")),
                float(row_match.group("c")),
            ]
        )
    if len(rows) < 3:
        return None
    return np.asarray(rows[:3], dtype=float)


def _format_values(values: Sequence[float]) -> str:
    if not values:
        raise ValueError("At least one value is required.")
    return " ".join(f"{value:.10g}" for value in values)


def _match_int(match: re.Match[str], base_name: str) -> int:
    value = match.group(base_name) or match.group(f"{base_name}_compact")
    return int(value)


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_ATOM_RE = re.compile(
    rf"(?:atom\s*:\s*(?P<atom>\d+)\s+typat\s*:\s*(?P<typat>\d+))|"
    rf"(?:Atom\s+(?P<atom_compact>\d+),\s*typat\s+(?P<typat_compact>\d+):"
    rf"\s*Cq\s*=\s*(?P<cq>{_FLOAT})\s*MHz\s+eta\s*=\s*(?P<eta>{_FLOAT}))"
)
_NUCLEAR_RE = re.compile(
    rf"Nuclear quad\. mom\. \(barns\)\s*:\s*(?P<q>{_FLOAT})\s+"
    rf"Cq \(MHz\)\s*:\s*(?P<cq>{_FLOAT})\s+eta\s*:\s*(?P<eta>{_FLOAT})"
)
_EIG_RE = re.compile(
    rf"efg eigval(?: \(au\))?\s*:\s*(?P<au>{_FLOAT})"
    rf"(?:\s*;\s*\(1\.0E\+21 V/m\^2\)\s*:\s*(?P<si>{_FLOAT}))?\s*"
    rf"(?:\r?\n)\s*-\s*eigvec\s*:\s*"
    rf"(?P<x>{_FLOAT})\s+(?P<y>{_FLOAT})\s+(?P<z>{_FLOAT})",
    re.MULTILINE,
)
