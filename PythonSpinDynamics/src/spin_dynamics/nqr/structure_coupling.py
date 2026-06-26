"""Dipole-dipole coupling estimates from small-molecule CIF structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import shlex

import numpy as np

from spin_dynamics.nqr.polarization_enhancement import PROTON_GAMMA_HZ_PER_T

MU0_OVER_4PI = 1.0e-7
PLANCK = 6.62607015e-34
ANGSTROM = 1.0e-10
GAMMA_14N_HZ_PER_T = 3.0766e6


@dataclass(frozen=True)
class CIFAtom:
    """One atom from a CIF atom-site loop."""

    label: str
    element: str
    fractional: np.ndarray


@dataclass(frozen=True)
class CIFStructure:
    """Minimal CIF crystal structure with atom sites and symmetry operations."""

    atoms: tuple[CIFAtom, ...]
    cell_lengths: tuple[float, float, float]
    cell_angles: tuple[float, float, float]
    symmetry_operations: tuple[str, ...]

    @property
    def cell_matrix(self) -> np.ndarray:
        """Return cell vectors as rows of a 3x3 matrix, in Angstrom."""

        a, b, c = self.cell_lengths
        alpha, beta, gamma = np.deg2rad(self.cell_angles)
        cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sin_g = np.sin(gamma)
        if abs(sin_g) < 1.0e-12:
            raise ValueError("cell gamma angle is singular")
        v_a = np.array([a, 0.0, 0.0], dtype=np.float64)
        v_b = np.array([b * cos_g, b * sin_g, 0.0], dtype=np.float64)
        cx = c * cos_b
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cz2 = c * c - cx * cx - cy * cy
        if cz2 < -1.0e-10:
            raise ValueError("cell parameters do not define a real lattice")
        v_c = np.array([cx, cy, np.sqrt(max(cz2, 0.0))], dtype=np.float64)
        return np.vstack([v_a, v_b, v_c])

    def atom(self, label: str) -> CIFAtom:
        """Return an atom by label."""

        for atom in self.atoms:
            if atom.label == label:
                return atom
        raise KeyError(f"unknown atom label: {label}")

    def cartesian(self, fractional: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to Cartesian Angstrom."""

        return np.asarray(fractional, dtype=np.float64) @ self.cell_matrix


@dataclass(frozen=True)
class ProtonDipolarCoupling:
    """One quadrupolar-nucleus to proton dipolar coupling estimate."""

    target_label: str
    proton_label: str
    proton_image: tuple[int, int, int]
    distance_angstrom: float
    vector_angstrom: np.ndarray
    coupling_hz: float
    secular_coupling_hz: float | None = None


@dataclass(frozen=True)
class ProtonCouplingEstimate:
    """Nearby-proton coupling summary for one quadrupolar nucleus."""

    target_label: str
    target_element: str
    proton_couplings: tuple[ProtonDipolarCoupling, ...]
    effective_rms_hz: float
    max_abs_hz: float
    sum_abs_hz: float


def load_cif_structure(path: str | Path) -> CIFStructure:
    """Load atom sites, cell parameters, and symmetry operations from a CIF."""

    text = Path(path).read_text(encoding="utf-8")
    items, loops = _parse_cif(text)
    atoms = _atom_sites_from_loops(loops)
    if not atoms:
        raise ValueError("CIF contains no atom-site coordinates")
    lengths = (
        _float_value(items["_cell_length_a"]),
        _float_value(items["_cell_length_b"]),
        _float_value(items["_cell_length_c"]),
    )
    angles = (
        _float_value(items.get("_cell_angle_alpha", "90")),
        _float_value(items.get("_cell_angle_beta", "90")),
        _float_value(items.get("_cell_angle_gamma", "90")),
    )
    sym_ops = _symmetry_operations_from_loops(loops)
    if not sym_ops:
        sym_ops = ("x,y,z",)
    return CIFStructure(
        atoms=tuple(atoms),
        cell_lengths=lengths,
        cell_angles=angles,
        symmetry_operations=tuple(sym_ops),
    )


def estimate_proton_dipolar_couplings_from_cif(
    path: str | Path,
    target_label: str,
    *,
    proton_radius_angstrom: float = 3.0,
    gamma_quadrupolar_hz_per_t: float = GAMMA_14N_HZ_PER_T,
    gamma_proton_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
    field_direction: Sequence[float] | None = None,
    max_periodic_images: int | None = None,
    max_results: int | None = None,
) -> ProtonCouplingEstimate:
    """Estimate nearby-proton dipolar couplings for ``target_label`` in a CIF.

    The scalar ``coupling_hz`` is the point-dipole prefactor
    ``mu0 h gamma_H gamma_Q / (4 pi r^3)``. If ``field_direction`` is supplied,
    ``secular_coupling_hz`` also includes the angular factor
    ``1 - 3 cos(theta)^2`` relative to that direction.
    """

    structure = load_cif_structure(path)
    return estimate_proton_dipolar_couplings(
        structure,
        target_label,
        proton_radius_angstrom=proton_radius_angstrom,
        gamma_quadrupolar_hz_per_t=gamma_quadrupolar_hz_per_t,
        gamma_proton_hz_per_t=gamma_proton_hz_per_t,
        field_direction=field_direction,
        max_periodic_images=max_periodic_images,
        max_results=max_results,
    )


def estimate_proton_dipolar_couplings(
    structure: CIFStructure,
    target_label: str,
    *,
    proton_radius_angstrom: float = 3.0,
    gamma_quadrupolar_hz_per_t: float = GAMMA_14N_HZ_PER_T,
    gamma_proton_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
    field_direction: Sequence[float] | None = None,
    max_periodic_images: int | None = None,
    max_results: int | None = None,
) -> ProtonCouplingEstimate:
    """Estimate nearby-proton dipolar couplings in a loaded CIF structure."""

    if proton_radius_angstrom <= 0.0 or not np.isfinite(proton_radius_angstrom):
        raise ValueError("proton_radius_angstrom must be positive and finite")
    if max_results is not None and int(max_results) < 1:
        raise ValueError("max_results must be positive")
    target = structure.atom(target_label)
    target_cart = structure.cartesian(target.fractional)
    direction = None if field_direction is None else _unit_vector(field_direction)

    image_range = _image_range(structure, proton_radius_angstrom, max_periodic_images)
    couplings: list[ProtonDipolarCoupling] = []
    seen: set[tuple[str, tuple[int, int, int], tuple[float, float, float]]] = set()
    for atom in structure.atoms:
        if atom.element.upper() != "H":
            continue
        for op in structure.symmetry_operations:
            frac = _apply_symmetry_operation(op, atom.fractional)
            for image in _translation_images(image_range):
                image_array = np.asarray(image, dtype=np.float64)
                proton_cart = structure.cartesian(frac + image_array)
                vector = proton_cart - target_cart
                distance = float(np.linalg.norm(vector))
                if distance < 1.0e-8 or distance > proton_radius_angstrom:
                    continue
                key = (
                    atom.label,
                    image,
                    tuple(np.round(vector, decimals=8)),
                )
                if key in seen:
                    continue
                seen.add(key)
                coupling = dipolar_coupling_hz(
                    distance,
                    gamma_a_hz_per_t=gamma_quadrupolar_hz_per_t,
                    gamma_b_hz_per_t=gamma_proton_hz_per_t,
                )
                secular = None
                if direction is not None:
                    cos_theta = float(np.dot(vector / distance, direction))
                    secular = coupling * (1.0 - 3.0 * cos_theta * cos_theta)
                couplings.append(
                    ProtonDipolarCoupling(
                        target_label=target.label,
                        proton_label=atom.label,
                        proton_image=image,
                        distance_angstrom=distance,
                        vector_angstrom=vector,
                        coupling_hz=coupling,
                        secular_coupling_hz=secular,
                    )
                )

    couplings.sort(key=lambda item: item.distance_angstrom)
    if max_results is not None:
        couplings = couplings[: int(max_results)]
    values = np.asarray([item.coupling_hz for item in couplings], dtype=np.float64)
    if values.size == 0:
        effective = max_abs = sum_abs = 0.0
    else:
        effective = float(np.sqrt(np.sum(values * values)))
        max_abs = float(np.max(np.abs(values)))
        sum_abs = float(np.sum(np.abs(values)))
    return ProtonCouplingEstimate(
        target_label=target.label,
        target_element=target.element,
        proton_couplings=tuple(couplings),
        effective_rms_hz=effective,
        max_abs_hz=max_abs,
        sum_abs_hz=sum_abs,
    )


def dipolar_coupling_hz(
    distance_angstrom: float,
    *,
    gamma_a_hz_per_t: float = GAMMA_14N_HZ_PER_T,
    gamma_b_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
) -> float:
    """Return the point-dipole coupling prefactor in Hz."""

    distance_m = float(distance_angstrom) * ANGSTROM
    if distance_m <= 0.0 or not np.isfinite(distance_m):
        raise ValueError("distance_angstrom must be positive and finite")
    return (
        MU0_OVER_4PI
        * PLANCK
        * abs(float(gamma_a_hz_per_t) * float(gamma_b_hz_per_t))
        / distance_m**3
    )


def _parse_cif(
    text: str,
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    tokens = _tokenize_cif(text)
    items: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "loop_":
            i += 1
            tags: list[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1
            values: list[str] = []
            while (
                i < len(tokens)
                and tokens[i] != "loop_"
                and not tokens[i].startswith("_")
            ):
                values.append(tokens[i])
                i += 1
            if tags:
                rows = [
                    values[j:j + len(tags)]
                    for j in range(0, len(values) - len(values) % len(tags), len(tags))
                ]
                loops.append((tags, rows))
            continue
        if token.startswith("_") and i + 1 < len(tokens):
            items[token] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return items, loops


def _tokenize_cif(text: str) -> list[str]:
    tokens: list[str] = []
    in_multiline = False
    multiline: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith(";"):
            if in_multiline:
                tokens.append("\n".join(multiline))
                multiline = []
                in_multiline = False
            else:
                in_multiline = True
            continue
        if in_multiline:
            multiline.append(raw_line)
            continue
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens.extend(list(lexer))
    if in_multiline:
        tokens.append("\n".join(multiline))
    return tokens


def _atom_sites_from_loops(
    loops: list[tuple[list[str], list[list[str]]]],
) -> list[CIFAtom]:
    for tags, rows in loops:
        if "_atom_site_label" not in tags:
            continue
        label_i = tags.index("_atom_site_label")
        element_i = _first_tag_index(tags, ["_atom_site_type_symbol"])
        fx_i = _first_tag_index(tags, ["_atom_site_fract_x"])
        fy_i = _first_tag_index(tags, ["_atom_site_fract_y"])
        fz_i = _first_tag_index(tags, ["_atom_site_fract_z"])
        if fx_i is None or fy_i is None or fz_i is None:
            continue
        atoms: list[CIFAtom] = []
        for row in rows:
            label = row[label_i]
            element = (
                row[element_i]
                if element_i is not None
                else _element_from_label(label)
            )
            atoms.append(
                CIFAtom(
                    label=label,
                    element=_normalize_element(element),
                    fractional=np.array(
                        [
                            _float_value(row[fx_i]),
                            _float_value(row[fy_i]),
                            _float_value(row[fz_i]),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
        return atoms
    return []


def _symmetry_operations_from_loops(
    loops: list[tuple[list[str], list[list[str]]]]
) -> list[str]:
    tags_to_try = [
        "_symmetry_equiv_pos_as_xyz",
        "_space_group_symop_operation_xyz",
    ]
    for tags, rows in loops:
        index = _first_tag_index(tags, tags_to_try)
        if index is not None:
            return [row[index] for row in rows]
    return []


def _first_tag_index(tags: list[str], names: list[str]) -> int | None:
    for name in names:
        if name in tags:
            return tags.index(name)
    return None


def _float_value(value: str) -> float:
    cleaned = str(value).strip().strip("'\"")
    if cleaned in ("?", "."):
        raise ValueError(f"missing numeric CIF value: {value!r}")
    if "(" in cleaned:
        cleaned = cleaned[: cleaned.index("(")]
    return float(cleaned)


def _normalize_element(value: str) -> str:
    text = str(value).strip().strip("'\"")
    if not text:
        return ""
    return text[0].upper() + text[1:].lower()


def _element_from_label(label: str) -> str:
    letters = "".join(ch for ch in str(label) if ch.isalpha())
    if len(letters) > 1 and letters[1].islower():
        return _normalize_element(letters[:2])
    return _normalize_element(letters[:1])


def _apply_symmetry_operation(operation: str, fractional: np.ndarray) -> np.ndarray:
    parts = [part.strip() for part in operation.strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise ValueError(f"unsupported symmetry operation: {operation!r}")
    x, y, z = np.asarray(fractional, dtype=np.float64)
    env = {"x": float(x), "y": float(y), "z": float(z)}
    return np.array(
        [_safe_eval_fractional(part, env) for part in parts],
        dtype=np.float64,
    )


def _safe_eval_fractional(expression: str, env: dict[str, float]) -> float:
    allowed = set("xyzXYZ0123456789+-*/(). ")
    if any(ch not in allowed for ch in expression):
        raise ValueError(f"unsupported symmetry expression: {expression!r}")
    return float(eval(expression.lower(), {"__builtins__": {}}, env))


def _image_range(
    structure: CIFStructure,
    radius_angstrom: float,
    max_periodic_images: int | None,
) -> int:
    if max_periodic_images is not None:
        return int(max_periodic_images)
    min_length = min(structure.cell_lengths)
    return max(1, int(np.ceil(radius_angstrom / min_length)) + 1)


def _translation_images(image_range: int):
    for i in range(-image_range, image_range + 1):
        for j in range(-image_range, image_range + 1):
            for k in range(-image_range, image_range + 1):
                yield (i, j, k)


def _unit_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("field_direction must be a nonzero finite vector")
    return vector / norm
