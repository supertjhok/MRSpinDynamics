"""Finite-strain EFG derivatives and quadrupolar drive couplings.

This module supports the piezoelectric-NQR workflow:

1. generate homogeneous ``+/-`` strain ABINIT EFG inputs;
2. collect the target-nucleus EFG tensors from completed ABINIT outputs;
3. finite-difference ``dV_ij / d epsilon_mu``;
4. convert those tensor derivatives into quadrupolar transition drive
   sensitivities in hertz per unit strain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex

import numpy as np

from .constants import BARN_M2, ELEMENTARY_CHARGE_C, PLANCK_CONSTANT_J_S
from .finite_displacement import (
    Crystal,
    abinit_input_with_positions,
    collect_efg_outputs,
)
from .tensors import EFGTensor


STRAIN_BASIS: dict[str, np.ndarray] = {
    "xx": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "yy": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "zz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "yz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    "xz": np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    "xy": np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}

ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8}
PSEUDO_BY_Z = {1: "H.xml", 6: "C.xml", 7: "N.xml", 8: "O.xml"}
QUADMOM_BY_Z = {1: 0.0, 6: 0.0, 7: 0.02044, 8: -0.02558}


@dataclass(frozen=True)
class StrainJob:
    """One homogeneous-strain EFG job."""

    name: str
    component: str
    sign: int
    strain_amplitude: float
    strain_tensor: np.ndarray
    crystal: Crystal

    def __post_init__(self) -> None:
        strain = np.asarray(self.strain_tensor, dtype=float).reshape(3, 3)
        if not np.allclose(strain, strain.T, atol=1e-14):
            raise ValueError("strain_tensor must be symmetric")
        object.__setattr__(self, "strain_tensor", strain)


@dataclass(frozen=True)
class StrainDerivative:
    """Finite-difference EFG derivative for one strain component."""

    component: str
    strain_amplitude: float
    derivative_si_per_strain: np.ndarray

    def __post_init__(self) -> None:
        derivative = np.asarray(self.derivative_si_per_strain, dtype=float)
        object.__setattr__(self, "derivative_si_per_strain", derivative.reshape(3, 3))


@dataclass(frozen=True)
class TransitionDriveCoupling:
    """Quadrupolar transition sensitivity for one strain derivative."""

    component: str
    lower: int
    upper: int
    frequency_hz: float
    cosine_hz_per_strain: float
    rwa_rabi_hz_per_strain: float


def strained_crystal(
    crystal: Crystal,
    strain_tensor: np.ndarray,
) -> Crystal:
    """Return an affinely strained crystal.

    Lattice vectors and Cartesian positions are transformed by ``I + epsilon``.
    Fractional coordinates are therefore preserved when the output is written
    back as ABINIT ``xred``.
    """

    strain = np.asarray(strain_tensor, dtype=float).reshape(3, 3)
    deformation = np.eye(3) + strain
    lattice = crystal.lattice_angstrom @ deformation.T
    cart = crystal.cart_angstrom @ deformation.T
    return Crystal(lattice, crystal.species_z, cart)


def generate_strain_jobs(
    crystal: Crystal,
    *,
    strain_amplitude: float = 1.0e-3,
    components: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy"),
) -> list[StrainJob]:
    """Build equilibrium plus central-difference homogeneous-strain jobs."""

    amplitude = float(strain_amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("strain_amplitude must be positive and finite")
    jobs = [
        StrainJob(
            name="equilibrium",
            component="equilibrium",
            sign=0,
            strain_amplitude=0.0,
            strain_tensor=np.zeros((3, 3)),
            crystal=crystal,
        )
    ]
    for component in components:
        if component not in STRAIN_BASIS:
            raise ValueError(f"unknown strain component: {component!r}")
        basis = STRAIN_BASIS[component]
        for sign, tag in ((+1, "plus"), (-1, "minus")):
            strain = sign * amplitude * basis
            jobs.append(
                StrainJob(
                    name=f"strain_{component}_{tag}",
                    component=component,
                    sign=sign,
                    strain_amplitude=amplitude,
                    strain_tensor=strain,
                    crystal=strained_crystal(crystal, strain),
                )
            )
    return jobs


def strain_manifest_dict(
    jobs: list[StrainJob],
    *,
    target_atom_index: int,
    structure_check: dict | None = None,
) -> dict:
    """Return a JSON-serializable strain-job manifest."""

    return {
        "kind": "homogeneous_strain_efg",
        "target_atom_index": int(target_atom_index),
        "structure_check": structure_check or {},
        "jobs": [
            {
                "name": job.name,
                "component": job.component,
                "sign": job.sign,
                "strain_amplitude": job.strain_amplitude,
                "strain_tensor": job.strain_tensor.tolist(),
            }
            for job in jobs
        ],
    }


def write_strain_jobs(
    base_input: str,
    jobs: list[StrainJob],
    output_dir: str | Path,
    *,
    target_atom_index: int,
    structure_check: dict | None = None,
) -> Path:
    """Write strained ABINIT inputs plus ``strain_manifest.json``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        text = abinit_input_with_lattice_and_positions(base_input, job.crystal)
        (directory / f"{job.name}.abi").write_text(text, encoding="utf-8")
    manifest = strain_manifest_dict(
        jobs,
        target_atom_index=target_atom_index,
        structure_check=structure_check,
    )
    (directory / "strain_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return directory


def abinit_input_with_lattice_and_positions(base_input: str, crystal: Crystal) -> str:
    """Return ``base_input`` with lattice and positions replaced."""

    text = _replace_lattice_block(base_input, crystal.lattice_angstrom)
    return abinit_input_with_positions(text, crystal)


def collect_strain_derivatives(
    manifest: dict,
    runs_dir: str | Path,
    *,
    target_atom_index: int | None = None,
    output_suffix: str = ".abo",
) -> tuple[EFGTensor, tuple[StrainDerivative, ...]]:
    """Parse strain-job outputs and finite-difference EFG derivatives."""

    efg_by_job = collect_efg_outputs(
        manifest,
        runs_dir,
        target_atom_index=target_atom_index,
        output_suffix=output_suffix,
    )
    equilibrium = efg_by_job["equilibrium"]
    derivatives: list[StrainDerivative] = []
    components = sorted(
        {
            job["component"]
            for job in manifest["jobs"]
            if job["component"] != "equilibrium"
        }
    )
    for component in components:
        plus = _manifest_job(manifest, component, +1)
        minus = _manifest_job(manifest, component, -1)
        delta = float(plus["strain_amplitude"])
        derivative = (
            efg_by_job[plus["name"]].matrix_si
            - efg_by_job[minus["name"]].matrix_si
        ) / (2.0 * delta)
        derivatives.append(
            StrainDerivative(
                component=component,
                strain_amplitude=delta,
                derivative_si_per_strain=derivative,
            )
        )
    return equilibrium, tuple(derivatives)


def quadrupole_hamiltonian_hz_from_efg(
    efg_matrix_si: np.ndarray,
    *,
    spin: float,
    quadrupole_moment_barns: float,
) -> np.ndarray:
    """Return the general quadrupole Hamiltonian divided by ``h`` in Hz."""

    efg = np.asarray(efg_matrix_si, dtype=float).reshape(3, 3)
    ops = _spin_matrices(spin)
    prefactor = (
        ELEMENTARY_CHARGE_C
        * quadrupole_moment_barns
        * BARN_M2
        / (2.0 * spin * (2.0 * spin - 1.0) * PLANCK_CONSTANT_J_S)
    )
    hamiltonian = np.zeros_like(ops[0], dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            hamiltonian += efg[i, j] * 0.5 * (ops[i] @ ops[j] + ops[j] @ ops[i])
    return prefactor * hamiltonian


def strain_transition_couplings(
    equilibrium_efg: EFGTensor,
    derivatives: tuple[StrainDerivative, ...] | list[StrainDerivative],
    *,
    spin: float = 1.0,
    quadrupole_moment_barns: float = 0.02044,
    transition_tolerance_hz: float = 1.0e-6,
) -> tuple[TransitionDriveCoupling, ...]:
    """Project strain derivatives onto quadrupolar transition matrix elements."""

    h0 = quadrupole_hamiltonian_hz_from_efg(
        equilibrium_efg.matrix_si,
        spin=spin,
        quadrupole_moment_barns=quadrupole_moment_barns,
    )
    values, vectors = np.linalg.eigh(h0)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]

    couplings: list[TransitionDriveCoupling] = []
    for derivative in derivatives:
        dh = quadrupole_hamiltonian_hz_from_efg(
            derivative.derivative_si_per_strain,
            spin=spin,
            quadrupole_moment_barns=quadrupole_moment_barns,
        )
        for lower in range(len(values)):
            for upper in range(lower + 1, len(values)):
                frequency = float(values[upper] - values[lower])
                if frequency <= transition_tolerance_hz:
                    continue
                matrix_element = vectors[:, lower].conj().T @ dh @ vectors[:, upper]
                cosine = float(abs(matrix_element))
                couplings.append(
                    TransitionDriveCoupling(
                        component=derivative.component,
                        lower=lower,
                        upper=upper,
                        frequency_hz=frequency,
                        cosine_hz_per_strain=cosine,
                        rwa_rabi_hz_per_strain=0.5 * cosine,
                    )
                )
    return tuple(couplings)


def cif_structure_metadata(path: str | Path) -> dict:
    """Read minimal CIF metadata used to guard glycine polymorph selection."""

    text = Path(path).read_text(encoding="utf-8")
    return {
        "path": str(path),
        "chemical_name_common": _cif_value(text, "_chemical_name_common"),
        "chemical_formula_sum": _cif_value(text, "_chemical_formula_sum"),
        "space_group": _cif_value(text, "_symmetry_space_group_name_H-M")
        or _cif_value(text, "_space_group_name_H-M_alt"),
        "cell_setting": _cif_value(text, "_symmetry_cell_setting"),
        "cell_volume": _cif_value(text, "_cell_volume"),
        "temperature": _cif_value(text, "_cell_measurement_temperature")
        or _cif_value(text, "_diffrn_ambient_temperature"),
        "ccdc": _cif_value(text, "_database_code_depnum_ccdc_archive"),
    }


def glycine_static_efg_input_from_cif(
    path: str | Path,
    *,
    ecut: float = 25.0,
    pawecutdg: float = 50.0,
    ngkpt: tuple[int, int, int] = (2, 2, 2),
    pseudo_dir: str = "Pseudodojo_paw_pw_standard",
    pawovlp: float | None = None,
    tbase_name: str = "glycine_beta_p21",
) -> tuple[str, list[dict]]:
    """Return a starter ABINIT static EFG input and expanded atom metadata.

    The bundled glycine CIF is in ``P 21`` with one molecule in the asymmetric
    unit.  The returned input expands the listed symmetry operations, disables
    symmetry in ABINIT, and keeps atom labels in comments so the target nitrogen
    can be chosen unambiguously.
    """

    text = Path(path).read_text(encoding="utf-8")
    items, loops = _parse_cif_tokens(text)
    atoms = _cif_atom_sites(loops)
    sym_ops = _cif_symmetry_operations(loops) or ("x,y,z",)
    lattice = _cif_lattice(items)
    expanded = _expand_cif_atoms(atoms, sym_ops)
    species = sorted({atom["z"] for atom in expanded})
    typat_by_z = {z: index + 1 for index, z in enumerate(species)}
    quadmom = [QUADMOM_BY_Z.get(z, 0.0) for z in species]
    pseudos = [PSEUDO_BY_Z.get(z, f"Z{z}.xml") for z in species]

    lines = [
        f"# Starter static EFG input generated from {Path(path).name}",
        "# Glycine CCDC 189379: non-centrosymmetric P 21 polymorph.",
        "# Converge ecut, pawecutdg, ngkpt, and relaxation before production use.",
        f"# job label: {tbase_name}",
        "",
        "nucefg 2",
        "quadmom " + " ".join(f"{value:.8g}" for value in quadmom),
        "",
        "acell 1.0 1.0 1.0 Angstrom",
        "rprim",
    ]
    lines.extend(f"  {row[0]:.12f}  {row[1]:.12f}  {row[2]:.12f}" for row in lattice)
    lines.extend(
        [
            "chkprim 0",
            "nsym 1",
            "",
            f"ntypat {len(species)}",
            "znucl " + " ".join(str(z) for z in species),
            f'pp_dirpath "$ABI_PSPDIR/{pseudo_dir}"',
            'pseudos "' + ", ".join(pseudos) + '"',
            "",
            f"natom {len(expanded)}",
            "typat",
        ]
    )
    typat_values = [typat_by_z[atom["z"]] for atom in expanded]
    lines.extend(_format_int_rows(typat_values))
    lines.append("")
    lines.append("xred")
    for index, atom in enumerate(expanded):
        frac = atom["fractional"]
        comment = (
            f"# {index:02d} {atom['label']} {atom['element']} "
            f"sym={atom['symmetry']}"
        )
        lines.append(
            f"  {frac[0]:.12f}  {frac[1]:.12f}  {frac[2]:.12f}  {comment}"
        )
    lines.extend(
        [
            "",
            "# Initial numerical values; converge before interpreting EFGs.",
            f"ecut {ecut:g}",
            f"pawecutdg {pawecutdg:g}",
            "ngkpt " + " ".join(str(int(v)) for v in ngkpt),
            "nshiftk 1",
            "shiftk 0.0 0.0 0.0",
            "",
            "occopt 1",
            "iscf 17",
            "nstep 80",
            "tolvrs 1.0d-14",
            "diemac 4.0",
            "",
            "prtden 0",
            "prtwf 0",
            "prteig 0",
            "prtvol 2",
            "",
        ]
    )
    if pawovlp is not None:
        lines.insert(-1, f"pawovlp {float(pawovlp):g}")
    return "\n".join(lines), expanded


def space_group_is_likely_centrosymmetric(space_group: str | None) -> bool:
    """Return a conservative centrosymmetry flag for common glycine groups."""

    if not space_group:
        return False
    normalized = re.sub(r"\s+", "", space_group.strip("'\"")).lower()
    centrosymmetric = {
        "p-1",
        "p21/c",
        "p21/n",
        "p2_1/c",
        "p2_1/n",
        "p21/a",
        "p2_1/a",
    }
    if normalized in centrosymmetric:
        return True
    return "/c" in normalized or "/n" in normalized or "-1" in normalized


def _replace_lattice_block(text: str, lattice_angstrom: np.ndarray) -> str:
    lattice = np.asarray(lattice_angstrom, dtype=float).reshape(3, 3)
    replaced_acell = _replace_keyword_values(
        text,
        "acell",
        3,
        "acell 1.0 1.0 1.0 Angstrom",
        allow_unit=True,
    )
    if replaced_acell is None:
        raise ValueError("base input has no acell block")
    rprim_block = _format_rprim_block(lattice)
    replaced = _replace_keyword_values(replaced_acell, "rprim", 9, rprim_block)
    if replaced is not None:
        return replaced
    match = re.search(r"(?:^|\n)[ \t]*acell\b[^\n]*(?:\n)?", replaced_acell)
    if match is None:
        raise ValueError("base input has no acell block")
    return (
        replaced_acell[: match.end()]
        + rprim_block
        + "\n"
        + replaced_acell[match.end() :]
    )


def _replace_keyword_values(
    text: str,
    keyword: str,
    value_count: int,
    new_block: str,
    *,
    allow_unit: bool = False,
) -> str | None:
    match = re.search(rf"(?:^|\n)([ \t]*){keyword}\b", text)
    if match is None:
        return None
    start = match.start() + (1 if text[match.start()] == "\n" else 0)
    token_re = re.compile(r"#[^\n]*|\S+")
    seen = 0
    end = match.end()
    while seen < value_count:
        token_match = token_re.search(text, end)
        if token_match is None:
            return None
        token = token_match.group(0)
        end = token_match.end()
        if token.startswith("#"):
            continue
        try:
            float(token.replace("d", "e").replace("D", "e"))
        except ValueError:
            return None
        seen += 1
    if allow_unit:
        token_match = token_re.search(text, end)
        if token_match is not None:
            token = token_match.group(0)
            if not token.startswith("#") and token.lower() in {
                "angstrom",
                "angstr",
                "angst",
                "ang",
                "bohr",
                "au",
            }:
                end = token_match.end()
    return text[:start] + new_block + "\n" + text[end:]


def _format_rprim_block(lattice_angstrom: np.ndarray) -> str:
    lines = ["rprim"]
    for row in lattice_angstrom:
        lines.append(f"  {row[0]:.12f}  {row[1]:.12f}  {row[2]:.12f}")
    return "\n".join(lines)


def _manifest_job(manifest: dict, component: str, sign: int) -> dict:
    for job in manifest["jobs"]:
        if job["component"] == component and int(job["sign"]) == sign:
            return job
    raise KeyError(f"missing manifest job for {component} sign {sign}")


def _spin_matrices(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    twice_spin = round(2.0 * spin)
    if not np.isclose(twice_spin, 2.0 * spin) or spin < 1.0:
        raise ValueError("spin must be an integer or half-integer >= 1")
    m_values = np.arange(spin, -spin - 1.0, -1.0)
    dim = len(m_values)
    iz = np.diag(m_values).astype(np.complex128)
    ip = np.zeros((dim, dim), dtype=np.complex128)
    im = np.zeros((dim, dim), dtype=np.complex128)
    index = {m: i for i, m in enumerate(m_values)}
    for m_value in m_values:
        raised = m_value + 1.0
        lowered = m_value - 1.0
        if raised in index:
            ip[index[raised], index[m_value]] = np.sqrt(
                spin * (spin + 1.0) - m_value * raised
            )
        if lowered in index:
            im[index[lowered], index[m_value]] = np.sqrt(
                spin * (spin + 1.0) - m_value * lowered
            )
    ix = 0.5 * (ip + im)
    iy = (ip - im) / (2.0j)
    return ix, iy, iz


def _cif_value(text: str, tag: str) -> str | None:
    match = re.search(rf"^{re.escape(tag)}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match is None:
        return None
    value = match.group(1).strip().strip("'\"")
    if value in {"?", "."}:
        return None
    return value


def _parse_cif_tokens(
    text: str,
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    tokens = _tokenize_cif(text)
    items: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index])
                index += 1
            values: list[str] = []
            while (
                index < len(tokens)
                and tokens[index] != "loop_"
                and not tokens[index].startswith("_")
            ):
                values.append(tokens[index])
                index += 1
            if tags:
                rows = [
                    values[start:start + len(tags)]
                    for start in range(
                        0,
                        len(values) - len(values) % len(tags),
                        len(tags),
                    )
                ]
                loops.append((tags, rows))
            continue
        if token.startswith("_") and index + 1 < len(tokens):
            items[token] = tokens[index + 1]
            index += 2
        else:
            index += 1
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


def _cif_lattice(items: dict[str, str]) -> np.ndarray:
    a = _float_cif(items["_cell_length_a"])
    b = _float_cif(items["_cell_length_b"])
    c = _float_cif(items["_cell_length_c"])
    alpha = np.deg2rad(_float_cif(items.get("_cell_angle_alpha", "90")))
    beta = np.deg2rad(_float_cif(items.get("_cell_angle_beta", "90")))
    gamma = np.deg2rad(_float_cif(items.get("_cell_angle_gamma", "90")))
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    v_a = np.array([a, 0.0, 0.0])
    v_b = np.array([b * cos_g, b * sin_g, 0.0])
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.vstack([v_a, v_b, [cx, cy, cz]])


def _cif_atom_sites(loops: list[tuple[list[str], list[list[str]]]]) -> list[dict]:
    for tags, rows in loops:
        if "_atom_site_label" not in tags:
            continue
        label_i = tags.index("_atom_site_label")
        fx_i = tags.index("_atom_site_fract_x")
        fy_i = tags.index("_atom_site_fract_y")
        fz_i = tags.index("_atom_site_fract_z")
        element_i = tags.index("_atom_site_type_symbol")
        atoms = []
        for row in rows:
            element = _normalize_element(row[element_i])
            atoms.append(
                {
                    "label": row[label_i],
                    "element": element,
                    "z": ELEMENT_Z[element],
                    "fractional": np.array(
                        [
                            _float_cif(row[fx_i]),
                            _float_cif(row[fy_i]),
                            _float_cif(row[fz_i]),
                        ],
                        dtype=float,
                    ),
                }
            )
        return atoms
    raise ValueError("CIF contains no atom-site loop")


def _cif_symmetry_operations(
    loops: list[tuple[list[str], list[list[str]]]]
) -> tuple[str, ...]:
    for tags, rows in loops:
        for tag in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
            if tag in tags:
                index = tags.index(tag)
                return tuple(row[index] for row in rows)
    return ()


def _expand_cif_atoms(atoms: list[dict], sym_ops: tuple[str, ...]) -> list[dict]:
    expanded: list[dict] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    for sym_index, operation in enumerate(sym_ops):
        for atom in atoms:
            frac = np.mod(_apply_symmetry_operation(operation, atom["fractional"]), 1.0)
            rounded = tuple(float(v) for v in np.round(frac, 10))
            key = (atom["label"], rounded)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                {
                    "label": atom["label"],
                    "element": atom["element"],
                    "z": atom["z"],
                    "fractional": frac,
                    "symmetry": sym_index + 1,
                }
            )
    return expanded


def _apply_symmetry_operation(operation: str, fractional: np.ndarray) -> np.ndarray:
    parts = [part.strip() for part in operation.strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise ValueError(f"unsupported symmetry operation: {operation!r}")
    x, y, z = np.asarray(fractional, dtype=float)
    env = {"x": float(x), "y": float(y), "z": float(z)}
    return np.array(
        [_safe_eval_fractional(part, env) for part in parts],
        dtype=float,
    )


def _safe_eval_fractional(expression: str, env: dict[str, float]) -> float:
    allowed = set("xyzXYZ0123456789+-*/(). ")
    if any(ch not in allowed for ch in expression):
        raise ValueError(f"unsupported symmetry expression: {expression!r}")
    return float(eval(expression.lower(), {"__builtins__": {}}, env))


def _float_cif(value: str) -> float:
    cleaned = str(value).strip().strip("'\"")
    if cleaned in {"?", "."}:
        raise ValueError(f"missing numeric CIF value: {value!r}")
    if "(" in cleaned:
        cleaned = cleaned[: cleaned.index("(")]
    return float(cleaned)


def _normalize_element(value: str) -> str:
    text = str(value).strip().strip("'\"")
    return text[0].upper() + text[1:].lower()


def _format_int_rows(values: list[int], columns: int = 12) -> list[str]:
    rows = []
    for start in range(0, len(values), columns):
        rows.append(
            "  " + " ".join(str(value) for value in values[start:start + columns])
        )
    return rows
