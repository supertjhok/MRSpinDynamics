"""GRAPE a slice-selective inversion by co-optimizing RF phase *and* the gradient.

Physical setup: a magnetic-field gradient turns position into off-resonance
(``offset = gradient * position``), so playing an RF pulse while a gradient is
on makes the inverted band localise to a slice. A plain rectangular pulse at a
*fixed* gradient inverts a broad, soft-edged band (a sinc-like ``Mz`` vs
position) and disturbs spins well outside the slice.

Here the gradient waveform ``g(t)`` is promoted to a genuine GRAPE control
channel (Milestone 2). The optimizer co-designs the per-segment RF **phase**
(amplitude held constant -- the primary mode for switching-power-amplifier
hardware) *and* the per-segment **gradient** to drive a per-position target --
inverted (``|down>``) inside the slice, untouched (``|up>``) outside -- across a
position ensemble. The single shared ``g(t)`` acts through a position-scaled
operator per ensemble member, exactly the ``positions @ gradient`` coupling the
motion/imaging engine uses (see ``workflows.slice_selective``).

Conditioning note: RF phase (radians, O(1)) and a raw gradient in hertz-per-
position (O(1e3)) are badly scale-mismatched for L-BFGS-B. We fold a fixed
characteristic offset scale into the gradient *operator*, so the optimized
gradient control is itself O(1) -- two well-conditioned control blocks. The
physical gradient in Hz/position is ``offset_scale * g(t)``.

This exercises the gradient control channel end to end: the position-ensemble
Hamiltonian batch, per-case targets, the joint phase+gradient control vector,
and a held-out position grid confirming the pulse sharpens the slice rather
than overfitting the training positions.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_state_numpy_grad,
)
from spin_dynamics.optimal_control.drivers import run_grape_multistart
from spin_dynamics.optimal_control.hamiltonians import (
    coupled_spin_control_model,
    gradient_control_operator,
    position_gradient_batch,
)
from spin_dynamics.optimal_control.parameterization import (
    constant_gradient_seed,
    rectangular_seed_phase,
)

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _mz_profile(model, hgrad_base, positions, amplitude, phase, gradient, dt):
    """<sigma_z> at each position after the RF+gradient pulse (from |up>)."""

    mz = np.zeros(positions.size, dtype=np.float64)
    for idx, pos in enumerate(positions):
        h_grad = float(pos) * hgrad_base
        psi = propagate_state_numpy_grad(
            model.h_drift, model.h_x, model.h_y, h_grad, amplitude, phase, gradient, dt, _PSI_UP
        )
        mz[idx] = np.real(np.conj(psi) @ _SIGMA_Z @ psi)
    return mz


def _selectivity_metrics(positions, mz, half_width):
    """In-slice inversion, out-of-slice preservation, and overall selectivity.

    ``selectivity`` is the mean per-position state-transfer fidelity to the
    slice target (invert inside, preserve outside) -- the held-out analogue of
    the training-ensemble objective, so it is the single scalar to compare.
    """

    inside = np.abs(positions) <= half_width
    outside = ~inside
    inversion = (1.0 - mz) / 2.0  # 1 when fully inverted (|down>)
    preservation = (1.0 + mz) / 2.0  # 1 when untouched (|up>)
    per_position = np.where(inside, inversion, preservation)
    return {
        "in_slice_inversion": float(np.mean(inversion[inside])),
        "out_slice_preservation": float(np.mean(preservation[outside])),
        "selectivity": float(np.mean(per_position)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=24, help="Number of pulse segments.")
    parser.add_argument("--nutation-hz", type=float, default=2000.0, help="Fixed RF nutation rate (Hz).")
    parser.add_argument("--slice-halfwidth", type=float, default=0.3, help="Slice half-width (position units).")
    parser.add_argument("--offset-scale-hz", type=float, default=6000.0,
                        help="Characteristic offset (Hz) a unit gradient*position produces.")
    parser.add_argument("--train-positions", type=int, default=15, help="Positions in the training ensemble.")
    parser.add_argument("--eval-points", type=int, default=81, help="Positions in the held-out evaluation grid.")
    parser.add_argument("--position-extent", type=float, default=1.0, help="Half-extent of the position grid.")
    parser.add_argument("--gradient-max", type=float, default=5.0, help="Gradient control cap (scaled units).")
    parser.add_argument("--starts", type=int, default=12, help="Random-phase restarts for the multistart optimizer.")
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
    half_width = float(args.slice_halfwidth)
    offset_scale = float(args.offset_scale_hz)

    # A rectangular constant-amplitude pulse inverts on-resonance when
    # 2*pi*nutation*T = pi, i.e. T = 1/(2*nutation): use that as the duration.
    total_duration_s = 0.5 / nutation_hz
    dt = np.full(n_segments, total_duration_s / n_segments)

    system = coupled_spin_system([0.0], [[0.0]])
    model = coupled_spin_control_model(system, include_gradient=True)
    # Gradient operator with the characteristic offset scale folded in, so the
    # optimized gradient control is O(1) alongside the O(1) RF phase.
    hgrad_base = offset_scale * gradient_control_operator(system)

    # ---- Fixed-gradient rectangular baseline ---------------------------------
    # Choose the fixed gradient so the slice edge sits near the nutation rate:
    # edge offset = g0 * half_width * offset_scale ~ nutation.
    g0 = nutation_hz / (half_width * offset_scale)
    baseline_phase = rectangular_seed_phase(n_segments)
    baseline_gradient = constant_gradient_seed(n_segments, gradient=g0)

    # ---- Training ensemble + per-position targets ----------------------------
    train_positions = np.linspace(-args.position_extent, args.position_extent, int(args.train_positions))
    hgrad_batch = position_gradient_batch(hgrad_base, train_positions)
    targets = np.stack([_PSI_DOWN if abs(p) <= half_width else _PSI_UP for p in train_positions])

    # Seed one start from the baseline; random phases for the rest. A perfectly
    # position-symmetric target with a constant-phase seed sits at a saddle
    # (mirrored offsets cancel in the mean gradient), so the random restarts do
    # the real work of escaping it.
    rng = np.random.default_rng(args.seed)
    random_phases = rng.uniform(-np.pi, np.pi, size=(max(args.starts - 1, 1), n_segments))
    initial_phases = np.vstack([baseline_phase[np.newaxis, :], random_phases])

    multistart = run_grape_multistart(
        model,
        n_segments,
        initial_phases=initial_phases,
        dt=dt,
        target=targets,
        psi0=_PSI_UP,
        optimize_amplitude=False,
        fixed_amplitude=nutation_hz,
        optimize_gradient=True,
        initial_gradient=baseline_gradient,
        gradient_max=float(args.gradient_max),
        gradient_operator_batch=hgrad_batch,
    )
    best = multistart.best_result

    # ---- Held-out evaluation -------------------------------------------------
    eval_positions = np.linspace(-args.position_extent, args.position_extent, int(args.eval_points))
    amp_fixed = np.full(n_segments, nutation_hz)
    base_mz = _mz_profile(model, hgrad_base, eval_positions, amp_fixed, baseline_phase, baseline_gradient, dt)
    opt_mz = _mz_profile(model, hgrad_base, eval_positions, amp_fixed, best.best_phase, best.best_gradient, dt)
    base_metrics = _selectivity_metrics(eval_positions, base_mz, half_width)
    opt_metrics = _selectivity_metrics(eval_positions, opt_mz, half_width)

    print("GRAPE slice-selective inversion (RF phase + gradient co-optimized)")
    print(f"segments: {n_segments}   nutation: {nutation_hz:.0f} Hz (fixed)   "
          f"pulse duration: {total_duration_s * 1e6:.0f} us   segment dt: {dt[0] * 1e6:.1f} us")
    print(f"slice half-width: {half_width:.3f}   offset scale: {offset_scale:.0f} Hz/unit   "
          f"baseline gradient g0: {g0:.3f}")
    print(f"training positions: {args.train_positions}   held-out grid: {args.eval_points}   "
          f"restarts: {args.starts}")
    print(f"optimizer iterations: {best.iterations}   success: {best.optimizer_success}")
    print()
    print(f"training-ensemble mean fidelity: baseline {best.initial_fidelity:.4f} "
          f"-> optimized {best.best_fidelity:.4f}")
    print("held-out profile metrics          baseline   optimized")
    print(f"  in-slice inversion (->1)        {base_metrics['in_slice_inversion']:.4f}     "
          f"{opt_metrics['in_slice_inversion']:.4f}")
    print(f"  out-slice preservation (->1)    {base_metrics['out_slice_preservation']:.4f}     "
          f"{opt_metrics['out_slice_preservation']:.4f}")
    print(f"  overall selectivity (->1)       {base_metrics['selectivity']:.4f}     "
          f"{opt_metrics['selectivity']:.4f}")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(eval_positions, base_mz, label="baseline (fixed grad, rect)", lw=1.5)
        axes[0].plot(eval_positions, opt_mz, label="GRAPE (phase + gradient)", lw=1.5)
        axes[0].axvspan(-half_width, half_width, color="gray", alpha=0.12)
        axes[0].axhline(-1.0, color="k", lw=0.5, ls=":")
        axes[0].axhline(1.0, color="k", lw=0.5, ls=":")
        axes[0].set_xlabel("position (slice units)")
        axes[0].set_ylabel(r"$\langle \sigma_z \rangle$")
        axes[0].set_title("Held-out inversion profile")
        axes[0].legend(fontsize=8)

        seg = np.arange(n_segments)
        axes[1].step(seg, np.mod(best.best_phase, 2 * np.pi), where="mid")
        axes[1].set_xlabel("segment index")
        axes[1].set_ylabel("RF phase (rad)")
        axes[1].set_title("Optimized phase (amplitude fixed)")

        axes[2].step(seg, baseline_gradient, where="mid", label="baseline grad")
        axes[2].step(seg, best.best_gradient, where="mid", label="GRAPE grad")
        axes[2].set_xlabel("segment index")
        axes[2].set_ylabel("gradient (scaled units)")
        axes[2].set_title("Gradient waveform")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
