"""Host-normalized performance regression gate for CI.

The orchestrator runs identical compact workloads from a Git baseline and the
candidate checkout in fresh Python processes on the same machine.  This removes
most runner-to-runner variation: only the candidate/baseline ratios are gated.

Examples::

    python -B benchmarks/regression_gate.py --baseline-ref HEAD^
    python -B benchmarks/regression_gate.py --baseline-ref origin/main --output .tmp/perf.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


@dataclass(frozen=True)
class RegressionPolicy:
    """Thresholds chosen to tolerate shared-runner noise but catch real regressions."""

    max_case_ratio: float = 1.50
    max_geomean_ratio: float = 1.25
    min_case_delta_seconds: float = 0.003


@dataclass(frozen=True)
class CaseComparison:
    """Candidate-versus-baseline result for one workload."""

    name: str
    baseline_seconds: float
    candidate_seconds: float
    ratio: float
    delta_seconds: float
    failed: bool


def compare_results(
    baseline: dict[str, float],
    candidate: dict[str, float],
    policy: RegressionPolicy,
) -> tuple[tuple[CaseComparison, ...], float, bool]:
    """Compare timing mappings and return cases, geometric mean, and failure."""

    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"benchmark cases differ; missing={missing}, extra={extra}")
    if not baseline:
        raise ValueError("benchmark result must contain at least one case")
    comparisons: list[CaseComparison] = []
    ratios: list[float] = []
    for name in sorted(baseline):
        before = float(baseline[name])
        after = float(candidate[name])
        if before <= 0.0 or after <= 0.0 or not math.isfinite(before + after):
            raise ValueError(f"benchmark case {name!r} has invalid timing")
        ratio = after / before
        delta = after - before
        failed = ratio > policy.max_case_ratio and delta > policy.min_case_delta_seconds
        comparisons.append(CaseComparison(name, before, after, ratio, delta, failed))
        ratios.append(
            max(1.0, ratio)
            if abs(delta) > policy.min_case_delta_seconds
            else 1.0
        )
    geomean = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    failed = any(case.failed for case in comparisons) or geomean > policy.max_geomean_ratio
    return tuple(comparisons), geomean, failed


def align_case_sets(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], tuple[str, ...]]:
    """Align evolving benchmark schemas without hiding removed coverage.

    Candidate-only cases are new coverage and cannot yet have a ratio. A case
    present in the baseline but absent from the candidate is an error because
    silently removing a benchmark would weaken the gate.
    """

    removed = sorted(set(baseline) - set(candidate))
    if removed:
        raise ValueError(f"candidate removed benchmark cases: {removed}")
    added = tuple(sorted(set(candidate) - set(baseline)))
    common = tuple(sorted(set(baseline) & set(candidate)))
    if not common:
        raise ValueError("baseline and candidate have no common benchmark cases")
    return (
        {name: baseline[name] for name in common},
        {name: candidate[name] for name in common},
        added,
    )


def _median_seconds(
    operation: Callable[[], Any], *, iterations: int, repeats: int, warmups: int
) -> float:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            for _ in range(iterations):
                operation()
            elapsed = time.perf_counter() - start
        finally:
            if enabled:
                gc.enable()
        samples.append(elapsed)
    return float(statistics.median(samples))


def _raw_kernel_case() -> Callable[[], Any]:
    import numpy as np

    from spin_dynamics.core.kernels import set_arb10_backend, sim_spin_dynamics_arb10
    from spin_dynamics.core.rotations import rf_matrix_elements

    numpts = 4001
    num_echoes = 96
    del_w = np.linspace(-10.0, 10.0, numpts)
    excitation = rf_matrix_elements(del_w, w1=1.0, tp=np.pi / 2.0, phi=0.0)
    refocusing = rf_matrix_elements(del_w, w1=1.0, tp=np.pi, phi=np.pi / 2.0)
    tp = [np.pi / 2.0]
    pulse = [1]
    amplitude = [1.0]
    acquire = [False]
    gradient = [0.0]
    for _ in range(num_echoes):
        tp.extend((2.0, np.pi, 2.0))
        pulse.extend((0, 2, 0))
        amplitude.extend((0.0, 1.0, 0.0))
        acquire.extend((False, False, True))
        gradient.extend((0.0, 0.0, 0.0))
    params = {
        "tp": np.asarray(tp),
        "pul": np.asarray(pulse),
        "amp": np.asarray(amplitude),
        "acq": np.asarray(acquire),
        "grad": np.asarray(gradient),
        "Rtot": [excitation, refocusing],
        "del_w": del_w,
        "del_wg": np.ones(numpts),
        "w_1": np.ones(numpts),
        "T1n": np.full(numpts, 1e6),
        "T2n": np.full(numpts, 1e6),
        "m0": np.ones(numpts, dtype=np.complex128),
        "mth": np.zeros(numpts, dtype=np.complex128),
    }
    set_arb10_backend("numpy")

    def run() -> Any:
        return sim_spin_dynamics_arb10(params)

    return run


def _workflow_case() -> Callable[[], Any]:
    from spin_dynamics.workflows import run_ideal_cpmg_train

    def run() -> Any:
        return run_ideal_cpmg_train(
            numpts=2001,
            maxoffs=10.0,
            num_echoes=32,
            t1_seconds=1.7,
            t2_seconds=1.1,
            num_workers=1,
            auto_refine_grid=False,
            rephase_action="ignore",
        )

    return run


def _sequence_compile_case() -> Callable[[], Any]:
    import numpy as np

    from spin_dynamics.sequences import (
        ADCEvent,
        GradientWaveform,
        RFPulse,
        SequenceBlock,
        SequenceIR,
        compile_sequence,
    )

    rf = RFPulse(np.ones(4), dwell_seconds=50e-6)
    gradient = GradientWaveform(np.linspace(0.0, 1000.0, 4), dwell_seconds=50e-6)
    adc = ADCEvent(4, dwell_seconds=50e-6, delay_seconds=300e-6)
    sequence = SequenceIR(
        tuple(
            SequenceBlock(
                duration_seconds=500e-6,
                rf=rf,
                gradients=(gradient, None, None),
                adc=adc,
                label=f"block_{index}",
            )
            for index in range(500)
        )
    )

    def run() -> Any:
        return compile_sequence(sequence)

    return run


def _spatial_sampling_case() -> Callable[[], Any]:
    import numpy as np

    from spin_dynamics.fields.interpolate import dlinear_sample

    axes = tuple(np.linspace(-1.0, 1.0, 28) for _ in range(3))
    x, y, z = np.meshgrid(*axes, indexing="ij")
    values = x + 2.0 * y - 0.5 * z
    rng = np.random.default_rng(42)
    positions = rng.uniform(-1.0, 1.0, size=(80_000, 3))

    def run() -> Any:
        return dlinear_sample(values, axes, positions)

    return run


def _powder_waveform_case() -> Callable[[], Any]:
    from spin_dynamics.nqr import (
        FieldDependentRelaxationModel,
        QuadrupolarSite,
        b0_b1_powder_average_halton,
        powder_carrier_frequency_hz,
        simulate_crossover_slse_powder,
    )

    site = QuadrupolarSite(
        spin=1.0,
        isotope="14N",
        quadrupole_frequency_hz=1.0e6,
        eta=0.2,
        gamma_hz_per_t=3.0766e6,
    )
    field = site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
    orientations = b0_b1_powder_average_halton(12)
    nutation_hz = 25.0e3
    carrier = powder_carrier_frequency_hz(
        site,
        field,
        orientations,
        nutation_hz=nutation_hz,
    )
    relaxation = FieldDependentRelaxationModel(
        temperature_kelvin=300.0,
        thermalization_time_seconds=0.1,
        dephasing_time_seconds=0.02,
    )

    def run() -> Any:
        return simulate_crossover_slse_powder(
            site,
            field,
            nutation_hz=nutation_hz,
            excitation_duration_seconds=10.0e-6,
            refocus_duration_seconds=20.0e-6,
            echo_spacing_seconds=200.0e-6,
            num_echoes=2,
            relaxation=relaxation,
            rf_frequency_hz=carrier,
            orientations=orientations,
            pulse_model="rwa",
            acquisition_duration_seconds=50.0e-6,
            acquisition_points=17,
            receiver_bandwidth_hz=100.0e3,
            num_workers=1,
            retain_local_results=False,
        )

    return run


def run_worker(source_root: Path, *, repeats: int, warmups: int) -> dict[str, float]:
    """Run the compact gate workloads against one package source tree."""

    source = str(source_root.resolve())
    sys.path.insert(0, source)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    builders = {
        "raw_arb10": (_raw_kernel_case, 2),
        "ideal_cpmg_workflow": (_workflow_case, 2),
        "sequence_compile": (_sequence_compile_case, 2),
        "spatial_sampling": (_spatial_sampling_case, 3),
        "nqr_powder_waveform": (_powder_waveform_case, 1),
    }
    cases: dict[str, tuple[Callable[[], Any], int]] = {}
    for name, (builder, iterations) in builders.items():
        try:
            cases[name] = (builder(), iterations)
        except (AttributeError, ImportError):
            # A candidate may add a benchmark for an API absent in an older
            # Git baseline. The orchestrator ratio-gates only the common set.
            continue
    return {
        name: _median_seconds(
            operation, iterations=iterations, repeats=repeats, warmups=warmups
        )
        for name, (operation, iterations) in cases.items()
    }


def _archive_source(ref: str, destination: Path) -> Path:
    command = [
        "git",
        "-C",
        str(REPOSITORY),
        "archive",
        "--format=tar",
        ref,
        "PythonSpinDynamics/src",
    ]
    archive = subprocess.run(command, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        # ``filter=`` is unavailable on the oldest supported Python versions.
        if sys.version_info >= (3, 12):
            handle.extractall(destination, filter="data")
        else:  # pragma: no cover - exercised by the Python 3.10/3.11 smoke matrix
            handle.extractall(destination)
    return destination / "PythonSpinDynamics" / "src"


def _worker_process(source_root: Path, repeats: int, warmups: int) -> dict[str, float]:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--repeats",
        str(repeats),
        "--warmups",
        str(warmups),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    line = next(line for line in reversed(completed.stdout.splitlines()) if line.strip())
    return {name: float(value) for name, value in json.loads(line).items()}


def _normalize_baseline_ref(value: str) -> str:
    ref = value.strip()
    if not ref or set(ref) == {"0"}:
        return "HEAD^"
    return ref


def _report(
    comparisons: tuple[CaseComparison, ...],
    geomean: float,
    policy: RegressionPolicy,
    added_cases: tuple[str, ...] = (),
) -> str:
    lines = [
        "Performance regression gate (candidate / Git baseline)",
        "case                      baseline    candidate      ratio  result",
    ]
    for case in comparisons:
        result = "FAIL" if case.failed else "pass"
        lines.append(
            f"{case.name:24s} {case.baseline_seconds:9.4f}s "
            f"{case.candidate_seconds:9.4f}s {case.ratio:9.3f}x  {result}"
        )
    aggregate = "FAIL" if geomean > policy.max_geomean_ratio else "pass"
    lines.append(f"geometric mean ratio: {geomean:.3f}x ({aggregate})")
    lines.append(
        f"limits: individual>{policy.max_case_ratio:.2f}x with "
        f">{policy.min_case_delta_seconds:.3f}s delta; "
        f"geometric mean>{policy.max_geomean_ratio:.2f}x"
    )
    if added_cases:
        lines.append(
            "new candidate cases (timed, not ratio-gated yet): "
            + ", ".join(added_cases)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD^")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-case-ratio", type=float, default=1.50)
    parser.add_argument("--max-geomean-ratio", type=float, default=1.25)
    parser.add_argument("--min-case-delta", type=float, default=0.003)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        raise SystemExit("repeats must be positive and warmups non-negative")
    if args.worker:
        if args.source_root is None:
            raise SystemExit("--worker requires --source-root")
        print(json.dumps(run_worker(args.source_root, repeats=args.repeats, warmups=args.warmups)))
        return

    policy = RegressionPolicy(
        max_case_ratio=float(args.max_case_ratio),
        max_geomean_ratio=float(args.max_geomean_ratio),
        min_case_delta_seconds=float(args.min_case_delta),
    )
    baseline_ref = _normalize_baseline_ref(args.baseline_ref)
    with tempfile.TemporaryDirectory(prefix="spin-dynamics-benchmark-") as temp:
        baseline_root = _archive_source(baseline_ref, Path(temp))
        baseline = _worker_process(baseline_root, args.repeats, args.warmups)
        candidate = _worker_process(ROOT / "src", args.repeats, args.warmups)
    aligned_baseline, aligned_candidate, added_cases = align_case_sets(
        baseline,
        candidate,
    )
    comparisons, geomean, failed = compare_results(
        aligned_baseline,
        aligned_candidate,
        policy,
    )
    print(_report(comparisons, geomean, policy, added_cases))
    if args.output is not None:
        payload = {
            "baseline_ref": baseline_ref,
            "policy": asdict(policy),
            "baseline": baseline,
            "candidate": candidate,
            "ratio_gated_cases": sorted(aligned_baseline),
            "new_candidate_cases": list(added_cases),
            "comparisons": [asdict(case) for case in comparisons],
            "geomean_ratio": geomean,
            "failed": failed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("performance regression threshold exceeded")


if __name__ == "__main__":
    main()
