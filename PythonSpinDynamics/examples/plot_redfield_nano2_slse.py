"""Plot NaNO2 14N SLSE decay from the shared Redfield/dipolar model.

The example reads the NaNO2 14N EFG parameters from the QuadrupolarDFT summary,
builds a spin-1 NQR site, estimates nearby 23Na/14N dipolar bath sources from
the NaNO2 CIF, and plots the coherent full-density-matrix SLSE echo-envelope
decay versus echo period.

Run with ``--output redfield_nano2_slse.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    OrientationSample,
    QuadrupolarSite,
    diagonalize_site,
    load_cif_structure,
    powder_average_grid,
    simulate_full_slse,
)
from spin_dynamics.nqr.structure_coupling import _apply_symmetry_operation  # noqa: E402
from spin_dynamics.relaxation import (  # noqa: E402
    DipolarRelaxationSource,
    RedfieldDipolarRelaxationModel,
    RigidSolidMotionalAveraging,
)


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"
NANO2_CIF = ROOT / "QuadrupolarDFT" / "structures" / "NaNO2" / "EntryWithCollCode82857.cif"

GAMMA_14N_HZ_PER_T = 3.0766e6
GAMMA_23NA_HZ_PER_T = 11.262e6


@dataclass(frozen=True)
class NaNO2Parameters:
    """NaNO2 14N quadrupolar parameters from the DFT summary."""

    cq_hz: float
    eta: float

    @property
    def nu_q_hz(self) -> float:
        """Return the spin-1 eta=0 simulator line frequency."""

        return 0.75 * abs(self.cq_hz)


@dataclass(frozen=True)
class SLSEDecaySweep:
    """Computed NaNO2 SLSE decay sweep."""

    label: str
    pulse_angle_degrees: float
    echo_spacing_seconds: np.ndarray
    detected_t2e_seconds: np.ndarray
    selected_echo_amplitudes: np.ndarray
    example_echo_times_seconds: np.ndarray
    example_echo_amplitudes: np.ndarray
    neighbor_distances_angstrom: np.ndarray
    neighbor_labels: tuple[str, ...]
    site: QuadrupolarSite
    transition_frequency_hz: float


@dataclass(frozen=True)
class SLSEComparison:
    """Single-orientation and powder-averaged NaNO2 SLSE sweeps."""

    single: SLSEDecaySweep
    powder: SLSEDecaySweep
    neighbor_distances_angstrom: np.ndarray
    neighbor_labels: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument("--correlation-us", type=float, default=1.0)
    parser.add_argument("--neighbor-radius-a", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--nutation-khz", type=float, default=25.0)
    parser.add_argument("--pulse-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--powder-angle-deg",
        type=float,
        default=119.0,
        help="Nominal powder SLSE pulse angle in degrees; spin-1 powder nutation peaks near 119.",
    )
    parser.add_argument("--powder-n-theta", type=int, default=6)
    parser.add_argument("--powder-n-phi", type=int, default=12)
    parser.add_argument("--min-spacing-us", type=float, default=120.0)
    parser.add_argument("--max-spacing-us", type=float, default=1400.0)
    parser.add_argument("--points", type=int, default=34)
    parser.add_argument("--num-echoes", type=int, default=48)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_nano2_parameters() -> NaNO2Parameters:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                return NaNO2Parameters(
                    cq_hz=float(row["mean_abs_cq_mhz"]) * 1.0e6,
                    eta=float(row["mean_eta"]),
                )
    raise RuntimeError(f"could not find NaNO2 14N parameters in {NANO2_SUMMARY}")


def _element_from_label(label: str) -> str:
    text = str(label)
    if text.startswith("Na"):
        return "Na"
    if text.startswith("N"):
        return "N"
    if text.startswith("O"):
        return "O"
    return text[:1]


def _nano2_dipolar_sources(
    *,
    radius_angstrom: float,
    max_neighbors: int,
) -> tuple[list[DipolarRelaxationSource], np.ndarray, tuple[str, ...]]:
    structure = load_cif_structure(NANO2_CIF)
    target = structure.atom("N1")
    target_cart = structure.cartesian(target.fractional)

    candidates: list[tuple[float, str, np.ndarray, float, float]] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    image_range = max(1, int(np.ceil(radius_angstrom / min(structure.cell_lengths))) + 1)
    for atom in structure.atoms:
        element = _element_from_label(atom.element)
        if element not in {"Na", "N"}:
            continue
        gamma = GAMMA_23NA_HZ_PER_T if element == "Na" else GAMMA_14N_HZ_PER_T
        spin = 1.5 if element == "Na" else 1.0
        for op in structure.symmetry_operations:
            frac = _apply_symmetry_operation(op, atom.fractional)
            for ia in range(-image_range, image_range + 1):
                for ib in range(-image_range, image_range + 1):
                    for ic in range(-image_range, image_range + 1):
                        image = np.array([ia, ib, ic], dtype=np.float64)
                        vector = structure.cartesian(frac + image) - target_cart
                        distance = float(np.linalg.norm(vector))
                        if distance < 1.0e-8 or distance > radius_angstrom:
                            continue
                        key = (atom.label, tuple(np.round(vector, decimals=8)))
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append((distance, atom.label, vector, gamma, spin))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:max_neighbors]
    sources = [
        DipolarRelaxationSource(
            vector_angstrom=vector,
            gamma_target_hz_per_t=GAMMA_14N_HZ_PER_T,
            gamma_bath_hz_per_t=gamma,
            bath_spin=spin,
        )
        for _, _, vector, gamma, spin in selected
    ]
    distances = np.array([item[0] for item in selected], dtype=np.float64)
    labels = tuple(item[1] for item in selected)
    return sources, distances, labels


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (4.0 * np.pi * nutation_hz)


def _echo_envelope_points(
    times: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a two-echo upper envelope to suppress even/odd modulation."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    magnitudes = np.abs(np.asarray(amplitudes, dtype=np.complex128).reshape(-1))
    if times.size != magnitudes.size:
        raise ValueError("times and amplitudes must have the same length")

    envelope_times: list[float] = []
    envelope_magnitudes: list[float] = []
    for start in range(0, magnitudes.size, 2):
        stop = min(start + 2, magnitudes.size)
        pair = magnitudes[start:stop]
        if pair.size == 0:
            continue
        idx = start + int(np.argmax(pair))
        envelope_times.append(float(times[idx]))
        envelope_magnitudes.append(float(magnitudes[idx]))
    return (
        np.asarray(envelope_times, dtype=np.float64),
        np.asarray(envelope_magnitudes, dtype=np.float64),
    )


def _fit_detected_t2e(
    times: np.ndarray,
    amplitudes: np.ndarray,
    *,
    use_upper_envelope: bool = False,
) -> float:
    """Fit a positive exponential time constant to the detected echo train."""

    if use_upper_envelope:
        fit_times, fit_magnitudes = _echo_envelope_points(times, amplitudes)
    else:
        fit_times = np.asarray(times, dtype=np.float64).reshape(-1)
        fit_magnitudes = np.abs(np.asarray(amplitudes, dtype=np.complex128).reshape(-1))
    usable = np.isfinite(fit_times) & np.isfinite(fit_magnitudes)
    usable &= fit_magnitudes > 0.0
    if np.count_nonzero(usable) < 2:
        return np.inf

    x = fit_times[usable]
    y = np.log(fit_magnitudes[usable])
    slope, _ = np.polyfit(x - x[0], y, deg=1)
    if slope >= -1.0e-12:
        return np.inf
    return float(-1.0 / slope)


def _run_sweep(
    *,
    site: QuadrupolarSite,
    transition_label: str,
    spacings: np.ndarray,
    nutation_hz: float,
    pulse_angle_degrees: float,
    num_echoes: int,
    orientations,
    relaxation: RedfieldDipolarRelaxationModel,
    label: str,
) -> SLSEDecaySweep:
    pulse_duration = _pulse_duration(pulse_angle_degrees, nutation_hz)
    transition = diagonalize_site(site).transition(transition_label)
    results = tuple(
        simulate_full_slse(
            site,
            nutation_hz=nutation_hz,
            rf_frequency_hz=transition.frequency_hz,
            excitation_duration_seconds=pulse_duration,
            refocus_duration_seconds=pulse_duration,
            echo_spacing_seconds=float(spacing),
            num_echoes=num_echoes,
            orientations=orientations,
            relaxation=relaxation,
        )
        for spacing in spacings
    )
    example = results[len(results) // 2]
    detected_t2e = np.array(
        [
            _fit_detected_t2e(result.echo_times, result.echo_amplitudes)
            for result in results
        ],
        dtype=np.float64,
    )
    return SLSEDecaySweep(
        label=label,
        pulse_angle_degrees=float(pulse_angle_degrees),
        echo_spacing_seconds=spacings,
        detected_t2e_seconds=detected_t2e,
        selected_echo_amplitudes=np.array(
            [result.echo_amplitudes[-1] for result in results],
            dtype=np.complex128,
        ),
        example_echo_times_seconds=example.echo_times,
        example_echo_amplitudes=example.echo_amplitudes,
        neighbor_distances_angstrom=np.empty(0, dtype=np.float64),
        neighbor_labels=(),
        site=site,
        transition_frequency_hz=transition.frequency_hz,
    )


def _simulate(args: argparse.Namespace) -> SLSEComparison:
    params = _load_nano2_parameters()
    site = QuadrupolarSite(
        spin=1.0,
        isotope="14N",
        label="NaNO2 N1",
        quadrupole_frequency_hz=params.nu_q_hz,
        eta=params.eta,
        gamma_hz_per_t=GAMMA_14N_HZ_PER_T,
    )
    sources, distances, labels = _nano2_dipolar_sources(
        radius_angstrom=float(args.neighbor_radius_a),
        max_neighbors=int(args.max_neighbors),
    )
    relaxation = RedfieldDipolarRelaxationModel.from_dipolar_sources(
        site.spin,
        sources,
        motion=RigidSolidMotionalAveraging(float(args.correlation_us) * 1.0e-6),
    )
    nutation_hz = float(args.nutation_khz) * 1.0e3
    spacings = np.linspace(
        float(args.min_spacing_us) * 1.0e-6,
        float(args.max_spacing_us) * 1.0e-6,
        int(args.points),
    )
    single = _run_sweep(
        site=site,
        transition_label=args.transition,
        spacings=spacings,
        nutation_hz=nutation_hz,
        num_echoes=int(args.num_echoes),
        pulse_angle_degrees=float(args.pulse_angle_deg),
        orientations=[OrientationSample((1.0, 0.0, 0.0))],
        relaxation=relaxation,
        label="single crystal",
    )
    powder = _run_sweep(
        site=site,
        transition_label=args.transition,
        spacings=spacings,
        nutation_hz=nutation_hz,
        num_echoes=int(args.num_echoes),
        pulse_angle_degrees=float(args.powder_angle_deg),
        orientations=powder_average_grid(
            int(args.powder_n_theta),
            int(args.powder_n_phi),
        ),
        relaxation=relaxation,
        label="powder average",
    )
    return SLSEComparison(
        single=single,
        powder=powder,
        neighbor_distances_angstrom=distances,
        neighbor_labels=labels,
    )


def _plot_one_row(axes, sim: SLSEDecaySweep) -> None:
    spacing_us = sim.echo_spacing_seconds * 1.0e6
    detected_ms = np.where(
        np.isfinite(sim.detected_t2e_seconds),
        sim.detected_t2e_seconds * 1.0e3,
        np.nan,
    )
    selected = np.abs(sim.selected_echo_amplitudes)
    selected /= np.max(selected) if np.max(selected) > 0 else 1.0

    axes[0].plot(spacing_us, detected_ms, marker="o", ms=3)
    axes[0].set_xlabel("SLSE echo period (us)")
    axes[0].set_ylabel("coherent-envelope T2e (ms)")
    axes[0].set_title(f"{sim.label}: coherent T2e")

    axes[1].plot(spacing_us, selected, color="tab:green")
    axes[1].set_xlabel("SLSE echo period (us)")
    axes[1].set_ylabel("normalized coherent echo")
    axes[1].set_title(f"{sim.pulse_angle_degrees:.0f} deg SLSE pulse")

    echo_times = sim.example_echo_times_seconds
    echo_magnitudes = np.abs(sim.example_echo_amplitudes)
    envelope_times, envelope_magnitudes = _echo_envelope_points(
        echo_times,
        sim.example_echo_amplitudes,
    )
    axes[2].semilogy(
        echo_times * 1.0e3,
        echo_magnitudes,
        marker="o",
        ms=3,
        label="coherent echoes",
    )
    axes[2].semilogy(
        envelope_times * 1.0e3,
        envelope_magnitudes,
        marker="s",
        ms=4,
        color="tab:orange",
        label="upper envelope",
    )
    fitted = _fit_detected_t2e(echo_times, sim.example_echo_amplitudes)
    if np.isfinite(fitted) and np.any(echo_magnitudes > 0.0):
        t0 = float(echo_times[0])
        y0 = float(echo_magnitudes[0])
        fit = y0 * np.exp(-(echo_times - t0) / fitted)
        axes[2].semilogy(
            echo_times * 1.0e3,
            fit,
            color="tab:red",
            lw=1.0,
            label="envelope fit",
        )
    axes[2].set_xlabel("echo time (ms)")
    axes[2].set_ylabel("echo magnitude")
    axes[2].set_title("mid-period echo train")
    axes[2].legend(frameon=False, fontsize="small")


def _plot(plt, comparison: SLSEComparison):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)
    _plot_one_row(axes[0], comparison.single)
    _plot_one_row(axes[1], comparison.powder)
    fig.suptitle(
        f"NaNO2 $^{{14}}$N Redfield SLSE, "
        f"{comparison.single.transition_frequency_hz / 1e6:.3f} MHz"
    )
    return fig


def _print_t2e_range(label: str, kind: str, values: np.ndarray) -> None:
    finite = np.isfinite(values)
    if np.any(finite):
        print(
            f"{label} {kind} T2e range: "
            f"{np.min(values[finite]) * 1e3:.4g} - "
            f"{np.max(values[finite]) * 1e3:.4g} ms"
        )


def main() -> None:
    args = _parse_args()
    if args.max_neighbors <= 0:
        raise SystemExit("--max-neighbors must be positive")
    if args.num_echoes <= 1:
        raise SystemExit("--num-echoes must be greater than one")
    if args.powder_n_theta <= 0 or args.powder_n_phi <= 0:
        raise SystemExit("--powder-n-theta and --powder-n-phi must be positive")
    plt = load_matplotlib(headless=args.output is not None)
    comparison = _simulate(args)
    print("NaNO2 14N Redfield SLSE")
    print(f"CIF: {NANO2_CIF}")
    print(f"DFT nu_Q: {comparison.single.site.quadrupole_frequency_hz / 1e6:.6g} MHz")
    print(f"eta: {comparison.single.site.eta:.6g}")
    print(
        f"transition {args.transition}: "
        f"{comparison.single.transition_frequency_hz / 1e6:.6g} MHz"
    )
    print(f"single-crystal SLSE pulse angle: {args.pulse_angle_deg:g} deg")
    print(f"powder SLSE pulse angle: {args.powder_angle_deg:g} deg")
    print("nearest stochastic bath sites:")
    for label, distance in zip(
        comparison.neighbor_labels,
        comparison.neighbor_distances_angstrom,
    ):
        print(f"  {label:>4s}  {distance:.3f} A")
    _print_t2e_range(
        "single-crystal",
        "coherent-envelope",
        comparison.single.detected_t2e_seconds,
    )
    _print_t2e_range("powder", "coherent-envelope", comparison.powder.detected_t2e_seconds)

    fig = _plot(plt, comparison)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
