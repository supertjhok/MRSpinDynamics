"""NQR database driven polarization-enhancement example.

This example demonstrates a full local-data workflow:

1. read 14N line frequencies and site metadata for a compound from
   ``NQRDatabase/data/exports/nqr.sqlite``;
2. estimate an effective 1H-14N dipolar coupling from a CIF structure;
3. run the adiabatic prepolarization-transfer model for each database
   transition.

The default uses glycine, whose NQR lines are present in the database and whose
CIF is bundled in ``QuadrupolarDFT/structures/Glycine``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "NQRDatabase" / "data" / "exports" / "nqr.sqlite"
DEFAULT_GLYCINE_CIF = (
    REPO_ROOT / "QuadrupolarDFT" / "structures" / "Glycine" / "189379.cif"
)


@dataclass(frozen=True)
class DatabaseTransition:
    label: str
    frequency_hz: float
    frequency_khz: float
    site_label: str | None
    qcc_khz: float | None
    eta: float | None
    t1_seconds: float | None
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseCompound:
    name: str
    formula: str | None
    conventional_formula: str | None
    transitions: tuple[DatabaseTransition, ...]
    protons_per_molecule: float
    nitrogens_per_molecule: float


@dataclass(frozen=True)
class DatabaseEnhancementResult:
    compound: DatabaseCompound
    transfer_result: object
    coupling_estimate: object | None
    coupling_hz: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull NQR transitions from NQRDatabase, estimate a CIF-based "
            "1H-14N coupling, and simulate prepolarization signal gains."
        )
    )
    parser.add_argument("--compound", default="Glycine",
                        help="Compound name to load from the NQR database.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB,
                        help="Path to NQRDatabase SQLite export.")
    parser.add_argument("--cif", type=Path, default=DEFAULT_GLYCINE_CIF,
                        help="CIF file used to estimate 1H-14N coupling.")
    parser.add_argument("--coupling-target", default="N1",
                        help="Quadrupolar atom label in the CIF, or 'auto'.")
    parser.add_argument("--coupling-radius", type=float, default=3.0,
                        help="Nearby-proton search radius in Angstrom.")
    parser.add_argument("--nh-coupling-hz", type=float,
                        help="Manual effective 1H-14N coupling rate (Hz).")
    parser.add_argument("--velocity", type=float, default=16.67,
                        help="Sample speed through crossings (cm/s).")
    parser.add_argument("--prepolarization", type=float, default=100.0,
                        help="Proton prepolarization time (s).")
    parser.add_argument("--t1h", type=float, default=48.6,
                        help="Proton T1 in the prepolarizing magnet (s).")
    parser.add_argument("--t1n", type=float,
                        help="14N retention time after transfer (s).")
    parser.add_argument("--proton-linewidth-khz", type=float, default=80.0,
                        help="Effective 1H linewidth near crossings (kHz).")
    parser.add_argument("--sample-length", type=float, default=20.0,
                        help="Sample length along motion axis (mm).")
    parser.add_argument("--sample-diameter", type=float, default=8.0,
                        help="Sample diameter (mm).")
    parser.add_argument("--axial-points", type=int, default=7,
                        help="Axial quadrature points for sample averaging.")
    parser.add_argument("--center-radius", type=float, default=25.4,
                        help="Halbach rod-center radius (mm).")
    parser.add_argument("--rod-width", type=float, default=25.4,
                        help="Square rod width (mm).")
    parser.add_argument("--magnet-length", type=float, default=101.6,
                        help="Magnet length along the transport axis (mm).")
    parser.add_argument("--remanence", type=float, default=1.15,
                        help="Rod remanence Br (T).")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Transport start coordinate z (mm).")
    parser.add_argument("--stop", type=float, default=100.0,
                        help="Transport stop coordinate z (mm).")
    parser.add_argument("--path-points", type=int, default=301,
                        help="Path samples for crossing detection.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show plot.")
    return parser.parse_args()


def load_compound_from_database(db_path: Path, compound_query: str) -> DatabaseCompound:
    if not db_path.exists():
        raise FileNotFoundError(f"NQR database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        compound = _find_compound(conn, compound_query)
        rows = conn.execute(
            """
            SELECT
                s.label AS sample_label,
                s.temperature_k,
                st.site_label,
                st.isotope,
                st.qcc_khz,
                st.eta,
                l.frequency_khz,
                l.transition_label,
                l.t1_s,
                l.source_id
            FROM samples s
            JOIN sites st ON st.sample_id = s.id
            JOIN lines l ON l.site_id = st.id
            WHERE s.compound_id = ?
              AND COALESCE(st.isotope, '') = '14N'
              AND l.frequency_khz IS NOT NULL
            ORDER BY l.frequency_khz, st.site_label, l.source_id
            """,
            [compound["id"]],
        ).fetchall()

    if not rows:
        raise ValueError(f"No 14N NQR lines found for {compound_query!r}")

    transitions = _deduplicate_transitions(rows)
    formula = compound["conventional_formula"] or compound["formula"]
    counts = parse_formula_counts(formula or "")
    protons = float(counts.get("H", 1))
    nitrogens = float(counts.get("N", 1))
    return DatabaseCompound(
        name=compound["canonical_name"],
        formula=compound["formula"],
        conventional_formula=compound["conventional_formula"],
        transitions=tuple(transitions),
        protons_per_molecule=max(protons, 1.0),
        nitrogens_per_molecule=max(nitrogens, 1.0),
    )


def _find_compound(conn: sqlite3.Connection, query: str) -> sqlite3.Row:
    exact = conn.execute(
        """
        SELECT id, canonical_name, formula, conventional_formula
        FROM compounds
        WHERE lower(canonical_name) = lower(?)
        LIMIT 1
        """,
        [query],
    ).fetchone()
    if exact is not None:
        return exact
    matches = conn.execute(
        """
        SELECT id, canonical_name, formula, conventional_formula
        FROM compounds
        WHERE lower(canonical_name) LIKE '%' || lower(?) || '%'
           OR lower(COALESCE(formula, '')) LIKE '%' || lower(?) || '%'
        ORDER BY length(canonical_name), canonical_name
        LIMIT 5
        """,
        [query, query],
    ).fetchall()
    if not matches:
        raise ValueError(f"No compound matching {query!r} found")
    return matches[0]


def _deduplicate_transitions(rows: list[sqlite3.Row]) -> list[DatabaseTransition]:
    grouped: dict[tuple[float, str, float | None, float | None], list[sqlite3.Row]] = {}
    for row in rows:
        key = (
            round(float(row["frequency_khz"]), 6),
            row["site_label"] or "",
            _rounded_optional(row["qcc_khz"]),
            _rounded_optional(row["eta"]),
        )
        grouped.setdefault(key, []).append(row)

    transitions: list[DatabaseTransition] = []
    for (_frequency, _site, _qcc, _eta), group in sorted(grouped.items()):
        first = group[0]
        frequency_khz = float(first["frequency_khz"])
        site_label = first["site_label"]
        label = first["transition_label"] or (
            f"{site_label or '14N'} {frequency_khz / 1e3:.3f} MHz"
        )
        t1_values = [
            float(row["t1_s"])
            for row in group
            if row["t1_s"] is not None and float(row["t1_s"]) > 0.0
        ]
        sources = tuple(sorted({row["source_id"] for row in group if row["source_id"]}))
        transitions.append(
            DatabaseTransition(
                label=label,
                frequency_hz=frequency_khz * 1e3,
                frequency_khz=frequency_khz,
                site_label=site_label,
                qcc_khz=first["qcc_khz"],
                eta=first["eta"],
                t1_seconds=(float(np.mean(t1_values)) if t1_values else None),
                source_ids=sources,
            )
        )
    return transitions


def _rounded_optional(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def parse_formula_counts(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element, count_text in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] = counts.get(element, 0) + int(count_text or "1")
    return counts


def resolve_coupling(args: argparse.Namespace) -> tuple[float, object | None]:
    if args.nh_coupling_hz is not None:
        return float(args.nh_coupling_hz), None
    if not args.cif.exists():
        return 1000.0, None
    from spin_dynamics.nqr import (
        estimate_proton_dipolar_couplings_from_cif,
        load_cif_structure,
    )

    target = args.coupling_target
    if target.lower() == "auto":
        structure = load_cif_structure(args.cif)
        target = next(
            (atom.label for atom in structure.atoms if atom.element.upper() == "N"),
            "",
        )
        if not target:
            return 1000.0, None
    estimate = estimate_proton_dipolar_couplings_from_cif(
        args.cif,
        target,
        proton_radius_angstrom=args.coupling_radius,
    )
    if estimate.effective_rms_hz > 0.0:
        return estimate.effective_rms_hz, estimate
    return 1000.0, estimate


def run_simulation(args: argparse.Namespace) -> DatabaseEnhancementResult:
    from spin_dynamics.nqr import (
        CylindricalSampleGeometry,
        HalbachPrepolarizationMagnet,
        LinearTransportMotion,
        PolarizationEnhancedNQRSample,
        simulate_adiabatic_polarization_transfer,
    )

    compound = load_compound_from_database(args.database, args.compound)
    coupling_hz, estimate = resolve_coupling(args)
    nitrogen_t1 = args.t1n or _median_database_t1(compound.transitions) or 5.0
    sample = PolarizationEnhancedNQRSample(
        name=compound.name,
        line_labels=tuple(item.label for item in compound.transitions),
        line_frequencies_hz=tuple(item.frequency_hz for item in compound.transitions),
        protons_per_molecule=compound.protons_per_molecule,
        nitrogens_per_molecule=compound.nitrogens_per_molecule,
        proton_t1_seconds=args.t1h,
        nitrogen_t1_seconds=nitrogen_t1,
        proton_linewidth_hz=args.proton_linewidth_khz * 1e3,
        proton_nitrogen_coupling_hz=coupling_hz,
    )
    geometry = CylindricalSampleGeometry(
        length=args.sample_length * 1e-3,
        diameter=args.sample_diameter * 1e-3,
        axial_points=args.axial_points,
        radial_rings=0,
    )
    magnet = HalbachPrepolarizationMagnet(
        center_radius=args.center_radius * 1e-3,
        length=args.magnet_length * 1e-3,
        remanence=args.remanence,
        rod_shape="square",
        rod_width=args.rod_width * 1e-3,
        rod_radius=0.5 * args.rod_width * 1e-3,
        n_cross=5,
        n_length=21,
    )
    motion = LinearTransportMotion(
        args.start * 1e-3,
        args.stop * 1e-3,
        velocity=args.velocity * 1e-2,
        axis="z",
    )
    result = simulate_adiabatic_polarization_transfer(
        magnet,
        sample,
        geometry,
        motion,
        prepolarization_time_seconds=args.prepolarization,
        path_points=args.path_points,
    )
    return DatabaseEnhancementResult(
        compound=compound,
        transfer_result=result,
        coupling_estimate=estimate,
        coupling_hz=coupling_hz,
    )


def _median_database_t1(transitions: tuple[DatabaseTransition, ...]) -> float | None:
    values = [item.t1_seconds for item in transitions if item.t1_seconds]
    if not values:
        return None
    return float(np.median(values))


def print_report(result: DatabaseEnhancementResult, args: argparse.Namespace) -> None:
    compound = result.compound
    transfer = result.transfer_result
    print("NQR database prepolarization example")
    print(f"  compound: {compound.name} ({compound.conventional_formula})")
    print(f"  database: {args.database}")
    print(f"  transitions loaded: {len(compound.transitions)}")
    print(
        "  molecule counts from formula: "
        f"H={compound.protons_per_molecule:g}, N={compound.nitrogens_per_molecule:g}"
    )
    print(f"  effective 1H-14N coupling: {result.coupling_hz:.1f} Hz")
    if result.coupling_estimate is not None:
        estimate = result.coupling_estimate
        print(
            f"  CIF target {estimate.target_label}: "
            f"{len(estimate.proton_couplings)} protons within "
            f"{args.coupling_radius:.1f} A"
        )
        for item in estimate.proton_couplings[:6]:
            print(
                f"    {item.proton_label:>4s} "
                f"r={item.distance_angstrom:.3f} A, "
                f"d={item.coupling_hz:.1f} Hz"
            )
    print(f"  center-field maximum on path: {np.max(transfer.b0_profile_tesla):.3f} T")
    print(f"  travel time: {transfer.travel_time_seconds:.3f} s")
    for transition, crossing, ratio, efficiency, ideal, practical in zip(
        compound.transitions,
        transfer.crossing_positions,
        transfer.adiabatic_ratios,
        transfer.transfer_efficiency,
        transfer.ideal_enhancement,
        transfer.practical_enhancement,
    ):
        sources = ",".join(transition.source_ids)
        print(
            f"  {transition.frequency_khz / 1e3:7.3f} MHz "
            f"({transition.site_label or '14N'}): "
            f"z={crossing * 1e3:5.1f} mm, "
            f"adiabatic ratio={ratio:.2f}, transfer={efficiency:.2f}, "
            f"ideal EF={ideal:.2f}, practical EF={practical:.2f}, "
            f"sources={sources}"
        )


def plot_result(plt, result: DatabaseEnhancementResult):
    compound = result.compound
    transfer = result.transfer_result
    labels = [f"{item.frequency_khz / 1e3:.3f}" for item in compound.transitions]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(transfer.b0_profile_positions * 1e3, transfer.b0_profile_tesla,
            color="k")
    for label, crossing, field in zip(
        labels, transfer.crossing_positions, transfer.crossing_fields_tesla
    ):
        ax.axhline(field, color="tab:blue", linestyle=":", linewidth=1.0)
        ax.axvline(crossing * 1e3, color="tab:orange", linestyle="--",
                   linewidth=1.0)
        ax.text(crossing * 1e3, field, f" {label} MHz", va="bottom")
    ax.set_xlabel("transport coordinate z (mm)")
    ax.set_ylabel("|B0| (T)")
    ax.set_title("Halbach fringe crossings from database lines")

    ax = axes[0, 1]
    width = 0.35
    ax.bar(x - width / 2, transfer.ideal_enhancement, width, label="ideal",
           color="0.75")
    ax.bar(x + width / 2, transfer.practical_enhancement, width,
           label="practical", color="tab:green")
    ax.set_xticks(x, labels)
    ax.set_xlabel("NQR transition (MHz)")
    ax.set_ylabel("signal enhancement factor")
    ax.set_title("Estimated signal gains")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.bar(x, transfer.transfer_efficiency, color="tab:purple")
    ax.set_xticks(x, labels)
    ax.set_xlabel("NQR transition (MHz)")
    ax.set_ylabel("crossing transfer efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Adiabatic crossing efficiency")

    ax = axes[1, 1]
    distances = []
    couplings = []
    proton_labels = []
    if result.coupling_estimate is not None:
        for item in result.coupling_estimate.proton_couplings[:8]:
            distances.append(item.distance_angstrom)
            couplings.append(item.coupling_hz)
            proton_labels.append(item.proton_label)
    if distances:
        ax.scatter(distances, couplings, color="tab:red")
        for distance, coupling, label in zip(distances, couplings, proton_labels):
            ax.text(distance, coupling, f" {label}", va="center")
    ax.set_xlabel("N-H distance (Angstrom)")
    ax.set_ylabel("point-dipole coupling (Hz)")
    ax.set_title("CIF-derived 1H-14N couplings")

    fig.suptitle(
        f"{compound.name}: database NQR lines plus CIF coupling estimate",
        fontsize=13,
    )
    return fig


def main() -> None:
    args = _parse_args()
    if args.path_points < 3:
        raise SystemExit("--path-points must be at least 3")
    if args.axial_points < 1:
        raise SystemExit("--axial-points must be at least 1")

    plt = load_matplotlib(headless=bool(args.output))
    result = run_simulation(args)
    print_report(result, args)
    fig = plot_result(plt, result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
