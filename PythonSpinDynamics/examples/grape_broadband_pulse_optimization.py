"""Optimize a phase-only GRAPE inversion pulse robust to B0 offset.

Physical setup: a plain rectangular (constant-phase) inversion pulse only
inverts spins near resonance -- spins offset by B0 inhomogeneity nutate about
a tilted effective field and end up far from the target state. Here GRAPE
optimizes the **phase** of each segment of a fixed-amplitude, fixed-duration
pulse (amplitude is held constant throughout -- the primary mode for
switching-power-amplifier hardware that cannot vary RF amplitude) to maximize
state-transfer fidelity to the inverted state, averaged over an ensemble of
B0 offsets (the robust/ensemble objective).

This exercises the new ``spin_dynamics.optimal_control`` GRAPE engine end to
end: Hamiltonian-model construction, the differentiable propagator chain, the
robust-ensemble fidelity objective, multistart optimization, and a held-out
evaluation grid (finer than -- and, for the worst-case check, wider than --
the training ensemble) to confirm the optimized pulse generalizes rather than
overfitting the exact training offsets.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_state_numpy,
)
from spin_dynamics.optimal_control.drivers import run_grape_multistart
from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model
from spin_dynamics.optimal_control.parameterization import rectangular_seed_phase

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)


def _model_for_offset(offset_hz: float):
    return coupled_spin_control_model(coupled_spin_system([offset_hz], [[0.0]]))


def _fidelity_vs_offset(
    phase: np.ndarray,
    amplitude_hz: float,
    dt: np.ndarray,
    offsets_hz: np.ndarray,
) -> np.ndarray:
    """Plain-NumPy evaluation (no autodiff needed) across a grid of offsets."""

    fidelities = np.zeros(offsets_hz.size, dtype=np.float64)
    amplitude = np.full(phase.size, amplitude_hz)
    for idx, offset_hz in enumerate(offsets_hz):
        model = _model_for_offset(float(offset_hz))
        psi_final = propagate_state_numpy(
            model.h_drift, model.h_x, model.h_y, amplitude, phase, dt, _PSI_UP
        )
        overlap = np.vdot(_PSI_DOWN, psi_final)
        fidelities[idx] = np.real(overlap * np.conj(overlap))
    return fidelities


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=16, help="Number of phase segments.")
    parser.add_argument("--nutation-hz", type=float, default=5000.0, help="Fixed RF nutation rate (Hz).")
    parser.add_argument(
        "--pulse-duration-us",
        type=float,
        default=200.0,
        help="Total pulse duration (us), split evenly across segments -- the "
        "segment duration this implies is the hardware control bandwidth: "
        "how fast the phase can actually be switched.",
    )
    parser.add_argument(
        "--train-maxoffs-hz",
        type=float,
        default=3500.0,
        help="Half-width (Hz) of the B0-offset ensemble used during optimization.",
    )
    parser.add_argument(
        "--train-offsets",
        type=int,
        default=7,
        help="Number of offsets in the training ensemble.",
    )
    parser.add_argument(
        "--eval-points",
        type=int,
        default=41,
        help="Number of offsets in the held-out evaluation grid (finer than training).",
    )
    parser.add_argument("--starts", type=int, default=24, help="Random restarts for the multistart optimizer.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for restarts.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save a summary figure.")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra (reverse-mode autodiff is the "
            "whole point of the engine). Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    n_segments = int(args.segments)
    nutation_hz = float(args.nutation_hz)
    total_duration_s = float(args.pulse_duration_us) * 1e-6
    dt = np.full(n_segments, total_duration_s / n_segments)

    train_offsets_hz = np.linspace(
        -args.train_maxoffs_hz, args.train_maxoffs_hz, int(args.train_offsets)
    )
    training_batch = [_model_for_offset(float(o)).h_drift for o in train_offsets_hz]
    model0 = _model_for_offset(float(train_offsets_hz[len(train_offsets_hz) // 2]))

    rect_phase = rectangular_seed_phase(n_segments)
    rng = np.random.default_rng(args.seed)
    random_starts = rng.uniform(
        -np.pi, np.pi, size=(max(args.starts - 1, 1), n_segments)
    )
    initial_phases = np.vstack([rect_phase[np.newaxis, :], random_starts])

    multistart = run_grape_multistart(
        model0,
        n_segments,
        initial_phases=initial_phases,
        dt=dt,
        target=_PSI_DOWN,
        psi0=_PSI_UP,
        fixed_amplitude=nutation_hz,
        hamiltonian_batch=training_batch,
        ensemble_reduction="mean",
    )
    best = multistart.best_result

    # Held-out evaluation grid: finer than, and as wide as, the training band.
    eval_offsets_hz = np.linspace(
        -args.train_maxoffs_hz, args.train_maxoffs_hz, int(args.eval_points)
    )
    rect_fidelity = _fidelity_vs_offset(rect_phase, nutation_hz, dt, eval_offsets_hz)
    opt_fidelity = _fidelity_vs_offset(best.best_phase, nutation_hz, dt, eval_offsets_hz)

    print("Phase-only GRAPE broadband inversion pulse (B0-offset-robust)")
    print(f"segments: {n_segments}   nutation rate: {nutation_hz:.0f} Hz (fixed, phase-only)")
    print(f"segment duration: {dt[0] * 1e6:.2f} us   total pulse duration: {total_duration_s * 1e6:.2f} us")
    print(f"training ensemble: {args.train_offsets} offsets in +/- {args.train_maxoffs_hz:.0f} Hz")
    print(f"restarts: {args.starts}   held-out evaluation grid: {args.eval_points} points")
    print()
    print(f"training-ensemble mean fidelity: rectangular {best.initial_fidelity:.6f} -> optimized {best.best_fidelity:.6f}")
    print(f"held-out mean fidelity        : rectangular {rect_fidelity.mean():.6f} -> optimized {opt_fidelity.mean():.6f}")
    print(f"held-out worst-case fidelity  : rectangular {rect_fidelity.min():.6f} -> optimized {opt_fidelity.min():.6f}")
    print(f"optimizer iterations: {best.iterations}   success: {best.optimizer_success}")
    print()
    print(
        "optimized phases (rad): "
        f"{np.array2string(np.mod(best.best_phase, 2 * np.pi), precision=3, separator=', ')}"
    )

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(eval_offsets_hz, rect_fidelity, label="rectangular", lw=1.5)
        axes[0].plot(eval_offsets_hz, opt_fidelity, label="GRAPE-optimized", lw=1.5)
        axes[0].axvspan(-args.train_maxoffs_hz, args.train_maxoffs_hz, color="gray", alpha=0.08)
        axes[0].set_xlabel("B0 offset (Hz)")
        axes[0].set_ylabel("inversion fidelity")
        axes[0].set_title("Held-out fidelity vs. B0 offset")
        axes[0].legend()

        segment_index = np.arange(n_segments)
        axes[1].step(segment_index, np.mod(best.best_phase, 2 * np.pi), where="mid")
        axes[1].set_xlabel("segment index")
        axes[1].set_ylabel("phase (rad)")
        axes[1].set_title("Optimized phase waveform (amplitude fixed)")

        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
