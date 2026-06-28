"""Powder spin-1 NQR SLSE in a weak static B0, accelerated by batched eigh.

A weak static field (an applied B0 or the residual field of a gradient) Zeeman-
splits the zero-field NQR line by a *different* amount for every crystallite
orientation, broadening the powder line. A spin-locked spin-echo (SLSE) train
refocuses that inhomogeneous Zeeman broadening, so its echo decay reports the
homogeneous T2 rather than the powder linewidth.

Simulating this means diagonalizing the quadrupole + Zeeman Hamiltonian once per
powder orientation. That per-orientation loop is exactly what the Phase 4 work
batches: ``diagonalize_sites_over_b0`` builds every orientation's Hamiltonian and
runs a single batched Hermitian eigensolve (``backend="numpy"`` over the stack,
or ``backend="jax"`` on GPU). ``simulate_slse`` / ``simulate_slse_offset_sweep``
now route their orientation scan through it.

This example (1) times the per-orientation loop against the batched solver for
the chosen powder grid, then (2) runs the SLSE offset sweep through the batched
path and reports the refocused echo train. See ``docs/performance.md``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    diagonalize_site,
    diagonalize_sites_over_b0,
    powder_average_grid,
    simulate_slse_offset_sweep,
)
from spin_dynamics.nqr._jax_eigh import JAX_AVAILABLE  # noqa: E402


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (2.0 * np.pi * nutation_hz)


def _orientation_b0_vectors(samples, b0_tesla: float) -> np.ndarray:
    vectors = []
    for sample in samples:
        direction = (
            sample.b0_direction_pas
            if sample.b0_direction_pas is not None
            else sample.b1_direction_pas
        )
        vectors.append(b0_tesla * np.asarray(direction, dtype=np.float64))
    return np.asarray(vectors, dtype=np.float64)


def _median_time(run, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Spin support: SLSE selective pulses currently support spin=1 only.",
    )
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument("--quadrupole-khz", type=float, default=900.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gamma-hz-per-t", type=float, default=3.077e6, help="14N gyromagnetic ratio.")
    parser.add_argument("--b0-mt", type=float, default=0.5, help="Weak static field in millitesla.")
    parser.add_argument("--nutation-khz", type=float, default=10.0)
    parser.add_argument("--pulse-angle", type=float, default=90.0)
    parser.add_argument("--echo-spacing-us", type=float, default=400.0)
    parser.add_argument("--num-echoes", type=int, default=16)
    parser.add_argument("--max-offset-khz", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--t2-ms", type=float, default=20.0)
    parser.add_argument("--n-theta", type=int, default=8)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "jax"],
        default="auto",
        help="Diagonalization backend; 'auto' uses jax when installed.",
    )
    args = parser.parse_args()

    backend = ("jax" if JAX_AVAILABLE else "numpy") if args.backend == "auto" else args.backend

    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
        gamma_hz_per_t=args.gamma_hz_per_t,
    )
    b0_tesla = args.b0_mt * 1e-3
    samples = powder_average_grid(args.n_theta, args.n_phi)
    b0_vectors = _orientation_b0_vectors(samples, b0_tesla)
    n_orient = b0_vectors.shape[0]

    # (1) Phase 4 demonstration: per-orientation loop vs one batched eigensolve.
    loop_time = _median_time(
        lambda: [diagonalize_site(site, vec) for vec in b0_vectors], args.repeats
    )
    if backend == "jax":  # absorb the one-time XLA compile
        diagonalize_sites_over_b0(site, b0_vectors, backend=backend)
    batched_time = _median_time(
        lambda: diagonalize_sites_over_b0(site, b0_vectors, backend=backend), args.repeats
    )

    # Parity: batched levels must match the per-orientation reference.
    batched = diagonalize_sites_over_b0(site, b0_vectors, backend=backend)
    max_level_diff = 0.0
    for idx, eig in enumerate(batched):
        ref = diagonalize_site(site, b0_vectors[idx])
        max_level_diff = max(max_level_diff, float(np.max(np.abs(eig.levels_hz - ref.levels_hz))))

    print("Powder spin-1 NQR SLSE in a weak B0 (Phase 4 batched diagonalization)")
    print(f"orientations: {n_orient}   B0: {args.b0_mt} mT   diagonalization backend: {backend}")
    speedup = loop_time / batched_time if batched_time else float("nan")
    print(
        f"diagonalize {n_orient} orientations: "
        f"per-orientation loop {loop_time * 1e3:.2f} ms -> batched {batched_time * 1e3:.2f} ms "
        f"({speedup:.1f}x)   max level diff {max_level_diff:.3e} Hz"
    )

    # (2) Run the SLSE offset sweep through the batched path.
    offsets_hz = np.linspace(-args.max_offset_khz * 1e3, args.max_offset_khz * 1e3, args.points)
    nutation_hz = args.nutation_khz * 1e3
    result = simulate_slse_offset_sweep(
        site,
        args.transition,
        offsets_hz,
        pulse_duration_seconds=_pulse_duration(args.pulse_angle, nutation_hz),
        nutation_hz=nutation_hz,
        echo_spacing_seconds=args.echo_spacing_us * 1e-6,
        num_echoes=args.num_echoes,
        orientations=samples,
        b0_tesla=b0_tesla,
        relaxation=NQRRelaxationModel(t1_seconds=np.inf, t2_seconds=args.t2_ms * 1e-3),
        backend=backend,
    )

    magnitude = np.abs(result.selected_echo_amplitudes)
    on_res = result.results[int(np.argmin(np.abs(offsets_hz)))]
    train = np.abs(on_res.echo_amplitudes)
    train = train / train[0] if train[0] else train
    print()
    print(f"SLSE offset sweep: {args.points} offsets x {args.num_echoes} echoes")
    print(f"peak selected-echo magnitude at offset {offsets_hz[int(np.argmax(magnitude))] / 1e3:.2f} kHz")
    print(
        "on-resonance echo train (normalized): "
        f"{np.array2string(train[: min(8, train.size)], precision=4, separator=', ')}"
    )
    print(
        f"cycle-derived T2eff at resonance: "
        f"{result.effective_t2eff_seconds[int(np.argmin(np.abs(offsets_hz)))] * 1e3:.2f} ms"
    )


if __name__ == "__main__":
    main()
