"""Finite-displacement driver for EFG mode curvatures.

The harmonic EFG average needs the second derivative of the EFG tensor along
each phonon mode, ``d^2 V_ij / dQ_k^2``.  These come from displacing the
structure along each mode by a small +/- step and recomputing the EFG with DFT.

Because the DFT runs are slow and happen outside this process, the driver
separates cleanly into:

1. **generate** -- from an equilibrium structure (parsed from a converged ABINIT
   input) and a set of phonon modes, write one displaced ABINIT input per
   ``+/- delta`` per mode, plus a ``manifest.json`` describing every job;
2. *(run ABINIT on each input locally)*;
3. **collect** -- parse the EFG of the target nucleus from each output and
   central-difference into a list of :class:`~quadrupolar_dft.vibrational.
   VibrationalMode`, ready for ``efg_temperature_sweep``.

Normal-coordinate convention: a phonon eigenvector ``eps`` is mass-weighted and
normalized (``sum |eps|^2 = 1``).  A normal-coordinate step ``delta_q`` (SI,
sqrt(kg) m) displaces atom ``i`` by ``delta_r_i = delta_q eps_i / sqrt(m_i)``,
matching the ``<Q^2>`` units of :func:`quadrupolar_dft.thermal.
mean_square_normal_coordinate`, so the resulting curvature has consistent units.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .abinit import parse_abinit_efg
from .constants import ANGSTROM_M, ATOMIC_MASS_UNIT_KG, BOHR_TO_ANGSTROM
from .tensors import EFGTensor
from .vibrational import VibrationalMode, efg_curvature_central_difference

# Atomic masses (amu) for the elements appearing in the workspace structures.
ATOMIC_MASS_AMU = {
    1: 1.008, 5: 10.811, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974, 16: 32.06,
    17: 35.45, 19: 39.098, 20: 40.078, 29: 63.546, 30: 65.38, 33: 74.922,
    35: 79.904, 53: 126.904,
}


@dataclass(frozen=True)
class Crystal:
    """A periodic structure: lattice and Cartesian positions, both in angstrom."""

    lattice_angstrom: np.ndarray
    species_z: tuple[int, ...]
    cart_angstrom: np.ndarray

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice_angstrom, dtype=float).reshape(3, 3)
        cart = np.asarray(self.cart_angstrom, dtype=float).reshape(-1, 3)
        if len(self.species_z) != cart.shape[0]:
            raise ValueError("species_z length must match the number of atoms")
        object.__setattr__(self, "lattice_angstrom", lattice)
        object.__setattr__(self, "species_z", tuple(int(z) for z in self.species_z))
        object.__setattr__(self, "cart_angstrom", cart)

    @property
    def natom(self) -> int:
        return self.cart_angstrom.shape[0]

    @property
    def masses_amu(self) -> np.ndarray:
        try:
            return np.array([ATOMIC_MASS_AMU[z] for z in self.species_z], dtype=float)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"no tabulated mass for Z={exc.args[0]}") from exc

    def with_positions(self, cart_angstrom: np.ndarray) -> "Crystal":
        return Crystal(self.lattice_angstrom, self.species_z, cart_angstrom)


@dataclass(frozen=True)
class PhononMode:
    """A harmonic mode: frequency and a mass-weighted, normalized eigenvector."""

    wavenumber_cm_inv: float
    eigenvector: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.eigenvector, dtype=float).reshape(-1, 3)
        norm = float(np.sqrt(np.sum(vector**2)))
        if not np.isclose(norm, 1.0, atol=1e-6):
            if norm == 0.0:
                raise ValueError("eigenvector must be non-zero")
            vector = vector / norm
        if float(self.wavenumber_cm_inv) <= 0.0:
            raise ValueError("wavenumber_cm_inv must be positive")
        object.__setattr__(self, "eigenvector", vector)


def cartesian_step_angstrom(
    mode: PhononMode, masses_amu: np.ndarray, delta_q_si: float
) -> np.ndarray:
    """Cartesian displacement (angstrom) for a normal-coordinate step ``delta_q``."""

    masses_kg = np.asarray(masses_amu, dtype=float) * ATOMIC_MASS_UNIT_KG
    delta_r_m = delta_q_si * mode.eigenvector / np.sqrt(masses_kg)[:, None]
    return delta_r_m / ANGSTROM_M


def normal_coordinate_step_si(
    mode: PhononMode,
    masses_amu: np.ndarray,
    *,
    max_displacement_angstrom: float,
) -> float:
    """Return ``delta_q`` (SI) giving a target maximum Cartesian displacement."""

    masses_kg = np.asarray(masses_amu, dtype=float) * ATOMIC_MASS_UNIT_KG
    per_atom = np.linalg.norm(mode.eigenvector, axis=1) / np.sqrt(masses_kg)
    largest = float(np.max(per_atom))
    if largest <= 0.0:
        raise ValueError("mode has no displacement")
    return (max_displacement_angstrom * ANGSTROM_M) / largest


@dataclass(frozen=True)
class DisplacementJob:
    """One structure to run: equilibrium (sign 0) or a +/- mode displacement."""

    name: str
    mode_index: int
    sign: int
    delta_q_si: float
    crystal: Crystal


def generate_displacement_jobs(
    crystal: Crystal,
    modes: list[PhononMode],
    *,
    max_displacement_angstrom: float = 0.05,
) -> list[DisplacementJob]:
    """Build the equilibrium job plus a +/- job per mode."""

    jobs = [
        DisplacementJob("equilibrium", -1, 0, 0.0, crystal),
    ]
    masses = crystal.masses_amu
    for index, mode in enumerate(modes):
        delta_q = normal_coordinate_step_si(
            mode, masses, max_displacement_angstrom=max_displacement_angstrom
        )
        step = cartesian_step_angstrom(mode, masses, delta_q)
        for sign in (+1, -1):
            displaced = crystal.with_positions(crystal.cart_angstrom + sign * step)
            tag = "plus" if sign > 0 else "minus"
            jobs.append(
                DisplacementJob(
                    name=f"mode{index:03d}_{tag}",
                    mode_index=index,
                    sign=sign,
                    delta_q_si=delta_q,
                    crystal=displaced,
                )
            )
    return jobs


def manifest_dict(jobs: list[DisplacementJob], *, target_atom_index: int) -> dict:
    """A JSON-serializable description of a job set."""

    return {
        "target_atom_index": int(target_atom_index),
        "jobs": [
            {
                "name": job.name,
                "mode_index": job.mode_index,
                "sign": job.sign,
                "delta_q_si": job.delta_q_si,
            }
            for job in jobs
        ],
    }


def write_jobs(
    base_input: str,
    jobs: list[DisplacementJob],
    output_dir: str | Path,
    *,
    target_atom_index: int,
) -> Path:
    """Write one ``<job>.abi`` per job and a ``manifest.json``; return the dir."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        text = abinit_input_with_positions(base_input, job.crystal)
        (directory / f"{job.name}.abi").write_text(text, encoding="utf-8")
    manifest = manifest_dict(jobs, target_atom_index=target_atom_index)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------------
# Collect: EFG outputs -> mode curvatures
# --------------------------------------------------------------------------


def vibrational_modes_from_efg(
    modes: list[PhononMode],
    jobs: list[DisplacementJob],
    efg_by_job: dict[str, EFGTensor],
) -> list[VibrationalMode]:
    """Central-difference per-mode EFG curvatures into VibrationalMode objects.

    ``efg_by_job`` maps a job name to the EFG tensor of the target nucleus from
    that run (including the ``"equilibrium"`` job).
    """

    equilibrium = efg_by_job["equilibrium"]
    by_mode_sign = {
        (job.mode_index, job.sign): job for job in jobs if job.mode_index >= 0
    }
    result: list[VibrationalMode] = []
    for index, mode in enumerate(modes):
        plus = by_mode_sign[(index, +1)]
        minus = by_mode_sign[(index, -1)]
        curvature = efg_curvature_central_difference(
            efg_by_job[minus.name],
            equilibrium,
            efg_by_job[plus.name],
            delta_q=plus.delta_q_si,
        )
        result.append(
            VibrationalMode(
                wavenumber_cm_inv=mode.wavenumber_cm_inv,
                efg_curvature_si=curvature,
                label=mode.label or f"mode{index:03d}",
            )
        )
    return result


def efg_tensor_from_record(record) -> EFGTensor:
    """Return the EFG tensor of one parsed ABINIT atom record.

    Prefers the printed total EFG tensor; otherwise reconstructs it from the
    eigenvalues and eigenvectors.
    """

    if record.tensor is not None:
        return record.tensor
    eigvals = np.asarray(record.eigvals_au, dtype=float)
    eigvecs = np.asarray(record.eigvecs, dtype=float)
    if eigvals.size != 3 or eigvecs.shape != (3, 3):
        raise ValueError("record lacks a total tensor and 3 eigenpairs")
    matrix = sum(
        value * np.outer(eigvecs[i], eigvecs[i]) for i, value in enumerate(eigvals)
    )
    return EFGTensor.from_components(matrix, unit="au")


def collect_efg_outputs(
    manifest: dict,
    runs_dir: str | Path,
    *,
    target_atom_index: int | None = None,
    output_suffix: str = ".abo",
) -> dict[str, EFGTensor]:
    """Parse each job's ABINIT output into the target-nucleus EFG tensor.

    Looks for ``<runs_dir>/<job_name><output_suffix>``.  ``target_atom_index`` is
    0-based (ABINIT atom indices are 1-based); defaults to the manifest value.
    """

    directory = Path(runs_dir)
    if target_atom_index is None:
        target_atom_index = int(manifest["target_atom_index"])
    abinit_atom = target_atom_index + 1

    efg_by_job: dict[str, EFGTensor] = {}
    for job in manifest["jobs"]:
        name = job["name"]
        path = directory / f"{name}{output_suffix}"
        if not path.exists():
            raise FileNotFoundError(f"missing ABINIT output for job {name!r}: {path}")
        records = parse_abinit_efg(path.read_text(encoding="utf-8"))
        match = next((r for r in records if r.atom_index == abinit_atom), None)
        if match is None:
            raise ValueError(
                f"no EFG record for atom {abinit_atom} in {path.name}"
            )
        efg_by_job[name] = efg_tensor_from_record(match)
    return efg_by_job


def vibrational_modes_from_collected(
    modes: list[PhononMode],
    manifest: dict,
    efg_by_job: dict[str, EFGTensor],
) -> list[VibrationalMode]:
    """Build mode curvatures from a manifest and collected EFG tensors."""

    equilibrium = efg_by_job["equilibrium"]
    by_mode_sign = {
        (job["mode_index"], job["sign"]): job
        for job in manifest["jobs"]
        if job["mode_index"] >= 0
    }
    result: list[VibrationalMode] = []
    for index, mode in enumerate(modes):
        plus = by_mode_sign[(index, +1)]
        minus = by_mode_sign[(index, -1)]
        curvature = efg_curvature_central_difference(
            efg_by_job[minus["name"]],
            equilibrium,
            efg_by_job[plus["name"]],
            delta_q=plus["delta_q_si"],
        )
        result.append(
            VibrationalMode(
                wavenumber_cm_inv=mode.wavenumber_cm_inv,
                efg_curvature_si=curvature,
                label=mode.label or f"mode{index:03d}",
            )
        )
    return result


# --------------------------------------------------------------------------
# ABINIT geometry input/output
# --------------------------------------------------------------------------

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
_LENGTH_UNITS = {
    "angstrom": 1.0, "angstr": 1.0, "angst": 1.0, "ang": 1.0,
    "bohr": BOHR_TO_ANGSTROM, "au": BOHR_TO_ANGSTROM, "atomic": BOHR_TO_ANGSTROM,
}


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _tokens_after(text: str, keyword: str) -> list[str]:
    match = re.search(rf"(?:^|\s){keyword}\b", text)
    if match is None:
        return []
    return text[match.end():].split()


def _read_floats(tokens: list[str], count: int) -> tuple[list[float], int]:
    values: list[float] = []
    index = 0
    while len(values) < count and index < len(tokens):
        token = tokens[index].replace("d", "e").replace("D", "e")
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(
                f"expected {count} numbers, got non-numeric token {tokens[index]!r}"
            ) from exc
        index += 1
    if len(values) < count:
        raise ValueError(f"expected {count} numbers, found {len(values)}")
    return values, index


def parse_abinit_structure(text: str) -> Crystal:
    """Parse acell/rprim/typat/znucl and xred or xcart from an ABINIT input."""

    clean = _strip_comments(text)

    natom = int(_tokens_after(clean, "natom")[0])
    ntypat = int(_tokens_after(clean, "ntypat")[0])
    znucl = [int(round(float(v))) for v in _read_floats(_tokens_after(clean, "znucl"), ntypat)[0]]
    typat = [int(round(float(v))) for v in _read_floats(_tokens_after(clean, "typat"), natom)[0]]
    species_z = [znucl[t - 1] for t in typat]

    acell_tokens = _tokens_after(clean, "acell")
    acell_values, consumed = _read_floats(acell_tokens, 3)
    acell_unit = 1.0
    if consumed < len(acell_tokens):
        unit_token = acell_tokens[consumed].lower()
        if unit_token in _LENGTH_UNITS:
            acell_unit = _LENGTH_UNITS[unit_token]
    acell = np.array(acell_values) * acell_unit

    rprim_tokens = _tokens_after(clean, "rprim")
    if rprim_tokens:
        rprim = np.array(_read_floats(rprim_tokens, 9)[0]).reshape(3, 3)
    else:
        rprim = np.eye(3)
    lattice = rprim * acell[:, None]

    xred_tokens = _tokens_after(clean, "xred")
    if xred_tokens:
        xred = np.array(_read_floats(xred_tokens, 3 * natom)[0]).reshape(natom, 3)
        cart = xred @ lattice
    else:
        for keyword, default_unit in (("xcart", BOHR_TO_ANGSTROM), ("xangst", 1.0)):
            tokens = _tokens_after(clean, keyword)
            if tokens:
                unit = default_unit
                offset = 0
                if tokens[0].lower() in _LENGTH_UNITS:
                    unit = _LENGTH_UNITS[tokens[0].lower()]
                    offset = 1
                values = _read_floats(tokens[offset:], 3 * natom)[0]
                cart = np.array(values).reshape(natom, 3) * unit
                break
        else:
            raise ValueError("input has neither xred, xcart, nor xangst")
    return Crystal(lattice, tuple(species_z), cart)


def _format_xred_block(crystal: Crystal) -> str:
    # Write fractional coordinates (xred): the most portable ABINIT position
    # format, and what the static EFG inputs already use. xred = cart @ inv(cell).
    xred = crystal.cart_angstrom @ np.linalg.inv(crystal.lattice_angstrom)
    lines = ["xred"]
    for row in xred:
        lines.append(f"  {row[0]:.12f}  {row[1]:.12f}  {row[2]:.12f}")
    return "\n".join(lines)


def abinit_input_with_positions(base_input: str, crystal: Crystal) -> str:
    """Return ``base_input`` with its position block replaced by displaced xangst.

    Everything else (cell, pseudopotentials, cutoffs, ``nucefg``/``quadmom``,
    k-points) is preserved verbatim so the displaced runs match the converged
    equilibrium settings.
    """

    natom = crystal.natom
    new_block = _format_xred_block(crystal)
    for keyword in ("xred", "xcart", "xangst"):
        replaced = _replace_position_block(base_input, keyword, natom, new_block)
        if replaced is not None:
            return replaced
    raise ValueError("base_input has no xred/xcart/xangst block to replace")


def _replace_position_block(
    text: str, keyword: str, natom: int, new_block: str
) -> str | None:
    """Replace ``keyword`` and its ``3*natom`` values, preserving other text."""

    match = re.search(rf"(?:^|\n)([ \t]*){keyword}\b", text)
    if match is None:
        return None
    start = match.start() + (1 if text[match.start()] == "\n" else 0)
    # Walk forward collecting numeric tokens (skipping comments and an optional
    # leading unit word) until 3*natom coordinates are consumed.
    pos = match.end()
    needed = 3 * natom
    seen = 0
    end = pos
    token_re = re.compile(r"#[^\n]*|\S+")
    first_word = True
    while seen < needed:
        token_match = token_re.search(text, end)
        if token_match is None:
            return None
        token = token_match.group(0)
        end = token_match.end()
        if token.startswith("#"):
            continue
        if first_word and token.lower() in _LENGTH_UNITS:
            first_word = False
            continue
        first_word = False
        normalized = token.replace("d", "e").replace("D", "e")
        try:
            float(normalized)
        except ValueError:
            return None
        seen += 1
    return text[:start] + new_block + "\n" + text[end:]
