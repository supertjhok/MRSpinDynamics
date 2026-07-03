"""Benchmark full density-matrix NQR FID/echo across spin 1 ... 9/2.

The full-model FID/echo sample free evolution at many time points; the sampler in
``spin_dynamics.nqr.full_dynamics`` diagonalizes the (constant) free generator once
and reconstructs every sample, instead of one matrix exponential per point. This
script times FID and echo for spin 1, 3/2, 5/2, 7/2, 9/2 (Hilbert dimension 3..10,
Liouville dimension 9..100) with a realistic sample count, so the cost of higher
spin -- and of the sampler -- is measurable.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    simulate_full_echo,
    simulate_full_fid,
)
from spin_dynamics.relaxation import NQRRelaxationModel  # noqa: E402


SPINS = (1.0, 1.5, 2.5, 3.5, 4.5)


def _time(callable_, repeats: int) -> float:
    best = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        best.append(time.perf_counter() - start)
    return statistics.median(best)


def _run(num_samples: int, repeats: int) -> list[dict[str, float]]:
    times = np.linspace(0.0, 500e-6, num_samples)
    relax = NQRRelaxationModel(t1_seconds=50e-3, t2_seconds=5e-3)
    rows: list[dict[str, float]] = []
    for spin in SPINS:
        site = QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1.0e6, eta=0.0)
        dim = site.dimension
        fid = _time(
            lambda site=site: simulate_full_fid(
                site, nutation_hz=10e3, pulse_duration_seconds=25e-6,
                times_seconds=times, rf_frequency_hz=1.0e6,
            ),
            repeats,
        )
        fid_relax = _time(
            lambda site=site: simulate_full_fid(
                site, nutation_hz=10e3, pulse_duration_seconds=25e-6,
                times_seconds=times, rf_frequency_hz=1.0e6, relaxation=relax,
            ),
            repeats,
        )
        echo = _time(
            lambda site=site: simulate_full_echo(
                site, nutation_hz=10e3, excitation_duration_seconds=25e-6,
                refocus_duration_seconds=50e-6, echo_spacing_seconds=520e-6,
                times_seconds=times, rf_frequency_hz=1.0e6,
            ),
            repeats,
        )
        rows.append({
            "spin": spin, "hilbert_dim": dim, "liouville_dim": dim * dim,
            "fid_ms": fid * 1e3, "fid_relax_ms": fid_relax * 1e3, "echo_ms": echo * 1e3,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = _run(args.num_samples, args.repeats)
    print(f"NQR full-model FID/echo, {args.num_samples} samples, "
          f"median of {args.repeats} runs (ms):")
    print(f"{'I':>4} {'dim':>4} {'L-dim':>6} {'FID':>8} {'FID+relax':>10} {'echo':>8}")
    for r in rows:
        print(f"{r['spin']:>4g} {r['hilbert_dim']:>4d} {r['liouville_dim']:>6d} "
              f"{r['fid_ms']:>8.2f} {r['fid_relax_ms']:>10.2f} {r['echo_ms']:>8.2f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
