"""Dipole-dipole coupling estimates from small-molecule CIF structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spin_dynamics.nqr import _cif
from spin_dynamics.nqr.polarization_enhancement import PROTON_GAMMA_HZ_PER_T

MU0_OVER_4PI = 1.0e-7
PLANCK = 6.62607015e-34
ANGSTROM = 1.0e-10
GAMMA_14N_HZ_PER_T = 3.0766e6
# Two symmetry-generated positions closer than this are the same site.
_SAME_SITE_ANGSTROM = 1.0e-4


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

        return _cif.cell_matrix(self.cell_lengths, self.cell_angles)

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
    items, loops = _cif.parse_cif(text)
    atoms = _atom_sites_from_loops(loops)
    if not atoms:
        raise ValueError("CIF contains no atom-site coordinates")
    lengths = (
        _cif.cif_float(items["_cell_length_a"]),
        _cif.cif_float(items["_cell_length_b"]),
        _cif.cif_float(items["_cell_length_c"]),
    )
    angles = (
        _cif.cif_float(items.get("_cell_angle_alpha", "90")),
        _cif.cif_float(items.get("_cell_angle_beta", "90")),
        _cif.cif_float(items.get("_cell_angle_gamma", "90")),
    )
    sym_ops = _cif.symmetry_operations_from_loops(loops)
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
    # Proton sites already counted, per label. A site is identified by its
    # position relative to the target only: different symmetry operations and
    # lattice images that land on the same point (a proton on a special
    # position) are one proton, so the image index must not be part of the key.
    seen: dict[str, list[np.ndarray]] = {}
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
                previous = seen.setdefault(atom.label, [])
                if any(
                    np.allclose(vector, other, rtol=0.0, atol=_SAME_SITE_ANGSTROM)
                    for other in previous
                ):
                    continue
                previous.append(vector)
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


def _atom_sites_from_loops(
    loops: list[tuple[list[str], list[list[str]]]],
) -> list[CIFAtom]:
    for tags, rows in loops:
        if "_atom_site_label" not in tags:
            continue
        label_i = tags.index("_atom_site_label")
        element_i = _cif.first_tag_index(tags, ["_atom_site_type_symbol"])
        fx_i = _cif.first_tag_index(tags, ["_atom_site_fract_x"])
        fy_i = _cif.first_tag_index(tags, ["_atom_site_fract_y"])
        fz_i = _cif.first_tag_index(tags, ["_atom_site_fract_z"])
        if fx_i is None or fy_i is None or fz_i is None:
            continue
        atoms: list[CIFAtom] = []
        for row in rows:
            label = row[label_i]
            element = (
                row[element_i]
                if element_i is not None
                else _cif.element_from_label(label)
            )
            atoms.append(
                CIFAtom(
                    label=label,
                    element=_cif.normalize_element(element),
                    fractional=np.array(
                        [
                            _cif.cif_float(row[fx_i]),
                            _cif.cif_float(row[fy_i]),
                            _cif.cif_float(row[fz_i]),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
        return atoms
    return []


# Kept as module attributes for existing importers (examples use it).
_apply_symmetry_operation = _cif.apply_symmetry_operation


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
