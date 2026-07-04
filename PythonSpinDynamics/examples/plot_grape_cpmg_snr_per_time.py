r"""GRAPE-optimized CPMG refocusing + axis-matched excitation (AMEX), SNR/time.

Two effects are explored together, both scored by the same figure of merit
already used for the SPA pulse catalog (``optimization.spa.evaluate_spa_metrics``):
``fom_time = echo_spacing / SNR**2`` (lower is better -- the time to reach a
target SNR by echo averaging), with ``echo_spacing = 2*t_free + t_pulse``. A
longer refocusing pulse buys (at best) a modest fidelity gain but directly
lengthens every echo, so it is penalized for generating fewer echoes per unit
time -- this example sweeps candidate refocusing-pulse durations and reports
where the trade-off actually optimizes, rather than assuming longer/fancier
pulses are always better.

The second effect is axis-matched excitation (AMEX): in an inhomogeneous
field, a CPMG-type refocusing cycle's *effective rotation axis* tilts away
from the nominal +x with offset. The steady state reached after many
refocusing cycles is the initial magnetization's projection onto that axis --
so exciting magnetization already aligned with the axis reaches the
asymptotic echo amplitude from the very first cycle (a flat echo train,
provably a fixed point of the cycle), while a conventional excitation must
"settle" through an oscillating transient that need not converge cleanly.

Methodology (matches the sequential/separate approach of the original AMEX
work, which this example also compares against a joint alternative): the
refocusing cycle is GRAPE-optimized first, assuming an idealized excitation
(instantaneous +x, no pulse-shape imperfection) -- the same assumption
``optimization.refocusing``'s "ideal" objectives make. Only once refocusing is
fixed does its effective axis exist to excite onto, so an excitation pulse is
then separately GRAPE-optimized (AMEX) to align with it. A joint
optimization (both pulses' phases together, directly maximizing the same
asymptotic ensemble signal) is also run at each duration for comparison --
answering, empirically, whether joint optimization leaves useful performance
on the table for this problem instance.

All pulses are phase-only (fixed peak amplitude/nutation rate) -- the primary
GRAPE mode for switching-power-amplifier hardware -- and every GRAPE result is
compared against a rectangular (constant-phase) pulse at the *same* peak B1
and *same* duration, never a stronger or longer one.

Panels: the optimized refocusing and excitation phase waveforms vs.
rectangular; the echo-train transient (+x-start vs. axis-aligned-start) for
both the GRAPE and rectangular refocusing cycles; the asymptotic time-domain
echo shape (GRAPE+AMEX vs. fully rectangular, same peak B1); and the
fom_time trade-off across the duration sweep. Run with ``--output figure.png``
to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system
from spin_dynamics.core.echo import calc_time_domain_echo_arb
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model
from spin_dynamics.optimal_control.parameterization import (
    phase_only_bounds,
    rectangular_seed_phase,
)
from spin_dynamics.optimization._bounded import scipy_maximize_with_grad

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import (
        iterate_unitary,
        propagate_state_scan,
        propagate_unitary_batched,
    )
    from spin_dynamics.optimal_control.objectives import (
        bloch_vector_to_ket,
        su2_effective_axis,
    )

TAU = 2.0 * np.pi
PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
PSI_X = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nutation-hz", type=float, default=5000.0, help="Peak B1, shared by every pulse compared.")
    parser.add_argument("--segment-us", type=float, default=8.0, help="Segment duration (hardware control bandwidth).")
    parser.add_argument("--ref-segments", type=int, nargs="+", default=[4, 8, 16], help="Candidate refocusing-pulse segment counts (duration sweep).")
    parser.add_argument("--exc-segments", type=int, default=4, help="Excitation-pulse segment count.")
    parser.add_argument("--free-precession-us", type=float, default=25.0, help="Fixed dead time on each side of the refocusing pulse.")
    parser.add_argument("--train-maxoffs-hz", type=float, default=3000.0, help="Half-width of the training B0-offset ensemble.")
    parser.add_argument("--train-offsets", type=int, default=7)
    parser.add_argument("--eval-points", type=int, default=41, help="Held-out evaluation grid size.")
    parser.add_argument("--n-cycles", type=int, default=25, help="Refocusing cycles simulated to reach the asymptotic regime.")
    parser.add_argument("--starts", type=int, default=10, help="Random restarts per optimization.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _transverse(psi):
    """<sigma_x> + i<sigma_y> = 2*conj(psi[0])*psi[1]."""

    return 2.0 * jnp.conj(psi[..., 0]) * psi[..., 1]


def _offset_batch(offsets_hz, h_x_ref):
    del h_x_ref
    return jnp.stack(
        [
            jnp.asarray(coupled_spin_control_model(coupled_spin_system([o], [[0.0]])).h_drift)
            for o in offsets_hz
        ]
    )


def _asymptotic_signal(u_batch, psi0_batch, n_cycles):
    """Ensemble-coherent transverse signal, averaged over the last 2 cycles."""

    traj = jax.vmap(iterate_unitary, in_axes=(0, 0, None))(u_batch, psi0_batch, n_cycles)
    return jnp.mean(_transverse(traj[:, -2:, :]))


def _multistart(value_and_grad_fn, n_params, *, starts, seed, rect_seed=None, bounds):
    rng = np.random.default_rng(seed)
    seeds = list(rng.uniform(-np.pi, np.pi, size=(max(starts - 1, 1), n_params)))
    if rect_seed is not None:
        seeds = [rect_seed] + seeds
    results = [
        scipy_maximize_with_grad(value_and_grad_fn, s, bounds=bounds.as_pairs())
        for s in seeds
    ]
    return max(results, key=lambda r: r.best_score)


def _make_refocusing_objective(h_x, h_y, h_drift_batch, w1_hz, n_ref, dt_ref, n_cycles):
    n_offsets = h_drift_batch.shape[0]
    psi0_batch = jnp.tile(jnp.asarray(PSI_X), (n_offsets, 1))

    def score(phi_ref):
        u_batch = propagate_unitary_batched(h_drift_batch, h_x, h_y, jnp.full(n_ref, w1_hz), phi_ref, dt_ref)
        signal = _asymptotic_signal(u_batch, psi0_batch, n_cycles)
        return jnp.real(signal * jnp.conj(signal))

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x):
        v, g = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g)

    return value_and_grad


def _make_amex_objective(h_x, h_y, h_drift_batch, target_batch, w1_hz, n_exc, dt_exc):
    def score(phi_exc):
        def one(h_drift_b, target_b):
            psi_f = propagate_state_scan(
                h_drift_b, h_x, h_y, jnp.full(n_exc, w1_hz), phi_exc, dt_exc, jnp.asarray(PSI_UP)
            )
            overlap = jnp.vdot(target_b, psi_f)
            return jnp.real(overlap * jnp.conj(overlap))

        return jnp.mean(jax.vmap(one)(h_drift_batch, target_batch))

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x):
        v, g = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g)

    return value_and_grad


def _make_joint_objective(h_x, h_y, h_drift_batch, w1_hz, n_exc, dt_exc, n_ref, dt_ref, n_cycles):
    n_offsets = h_drift_batch.shape[0]

    def score(x):
        phi_exc, phi_ref = x[:n_exc], x[n_exc:]

        def excite(h_drift_b):
            return propagate_state_scan(
                h_drift_b, h_x, h_y, jnp.full(n_exc, w1_hz), phi_exc, dt_exc, jnp.asarray(PSI_UP)
            )

        psi0_batch = jax.vmap(excite)(h_drift_batch)
        u_batch = propagate_unitary_batched(h_drift_batch, h_x, h_y, jnp.full(n_ref, w1_hz), phi_ref, dt_ref)
        signal = _asymptotic_signal(u_batch, psi0_batch, n_cycles)
        return jnp.real(signal * jnp.conj(signal))

    _vg = jax.jit(jax.value_and_grad(score))

    def value_and_grad(x):
        v, g = _vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g)

    return value_and_grad, n_offsets


@dataclass(frozen=True)
class DurationResult:
    n_ref: int
    t_pulse_s: float
    echo_spacing_s: float
    phi_ref: np.ndarray
    phi_exc_amex: np.ndarray
    phi_joint: np.ndarray
    snr_rect_rect: float
    snr_grape_ideal: float
    snr_grape_amex: float
    snr_grape_rectexc: float
    snr_joint: float

    def fom_time(self, snr: float) -> float:
        return self.echo_spacing_s / snr**2


def main() -> None:
    args = _parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra (reverse-mode autodiff is the "
            "whole point of the engine). Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    w1_hz = float(args.nutation_hz)
    dt_seg = float(args.segment_us) * 1e-6
    t_free = float(args.free_precession_us) * 1e-6
    n_exc = int(args.exc_segments)
    dt_exc = jnp.full(n_exc, dt_seg)

    train_offsets = np.linspace(-args.train_maxoffs_hz, args.train_maxoffs_hz, int(args.train_offsets))
    eval_offsets = np.linspace(-args.train_maxoffs_hz, args.train_maxoffs_hz, int(args.eval_points))

    m0 = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    h_x, h_y = jnp.asarray(m0.h_x), jnp.asarray(m0.h_y)
    train_batch = _offset_batch(train_offsets, h_x)
    eval_batch = _offset_batch(eval_offsets, h_x)

    print("GRAPE CPMG refocusing + axis-matched excitation (AMEX), SNR per unit time")
    print(f"peak nutation rate: {w1_hz:.0f} Hz (shared by every pulse)   segment: {dt_seg * 1e6:.1f} us")
    print(f"training ensemble: {args.train_offsets} offsets in +/- {args.train_maxoffs_hz:.0f} Hz, {args.n_cycles} refocusing cycles")
    print()

    duration_results: list[DurationResult] = []
    for n_ref in args.ref_segments:
        n_ref = int(n_ref)
        dt_ref = jnp.full(n_ref, dt_seg)
        t_pulse = n_ref * dt_seg
        echo_spacing = 2.0 * t_free + t_pulse
        bounds_ref = phase_only_bounds(n_ref)
        rect_ref = rectangular_seed_phase(n_ref)

        # 1. GRAPE refocusing, assuming ideal (instantaneous) +x excitation.
        vg_ref = _make_refocusing_objective(h_x, h_y, train_batch, w1_hz, n_ref, dt_ref, args.n_cycles)
        best_ref = _multistart(vg_ref, n_ref, starts=args.starts, seed=args.seed, rect_seed=rect_ref, bounds=bounds_ref)
        phi_ref = best_ref.best_x

        # Evaluate on the held-out grid: ideal +x, and rectangular refocusing baseline.
        u_eval = propagate_unitary_batched(eval_batch, h_x, h_y, jnp.full(n_ref, w1_hz), jnp.asarray(phi_ref), dt_ref)
        psi_x_eval = jnp.tile(jnp.asarray(PSI_X), (eval_offsets.size, 1))
        snr_grape_ideal = float(jnp.abs(_asymptotic_signal(u_eval, psi_x_eval, args.n_cycles)))

        u_rect_eval = propagate_unitary_batched(eval_batch, h_x, h_y, jnp.full(n_ref, w1_hz), jnp.asarray(rect_ref), dt_ref)

        # 2. AMEX: axis from the GRAPE refocusing cycle (held-out grid), excite onto it.
        axes, _angles = jax.vmap(su2_effective_axis)(u_eval)
        targets = jax.vmap(bloch_vector_to_ket)(axes)
        bounds_exc = phase_only_bounds(n_exc)
        rect_exc = rectangular_seed_phase(n_exc)

        vg_amex = _make_amex_objective(h_x, h_y, eval_batch, targets, w1_hz, n_exc, dt_exc)
        best_amex = _multistart(vg_amex, n_exc, starts=args.starts, seed=args.seed, rect_seed=rect_exc, bounds=bounds_exc)
        phi_exc_amex = best_amex.best_x

        def excite_all(phi_exc):
            def one(h_drift_b):
                return propagate_state_scan(h_drift_b, h_x, h_y, jnp.full(n_exc, w1_hz), jnp.asarray(phi_exc), dt_exc, jnp.asarray(PSI_UP))
            return jax.vmap(one)(eval_batch)

        psi0_amex = excite_all(phi_exc_amex)
        snr_grape_amex = float(jnp.abs(_asymptotic_signal(u_eval, psi0_amex, args.n_cycles)))

        psi0_rectexc = excite_all(rect_exc)
        snr_grape_rectexc = float(jnp.abs(_asymptotic_signal(u_eval, psi0_rectexc, args.n_cycles)))

        snr_rect_rect = float(jnp.abs(_asymptotic_signal(u_rect_eval, psi0_rectexc, args.n_cycles)))

        # 3. Joint optimization: excitation + refocusing together, same headline metric.
        vg_joint, _ = _make_joint_objective(h_x, h_y, train_batch, w1_hz, n_exc, dt_exc, n_ref, dt_ref, args.n_cycles)
        bounds_joint = phase_only_bounds(n_exc + n_ref)
        rect_joint = np.concatenate([rect_exc, rect_ref])
        best_joint = _multistart(vg_joint, n_exc + n_ref, starts=args.starts, seed=args.seed, rect_seed=rect_joint, bounds=bounds_joint)
        phi_joint = best_joint.best_x
        phi_joint_exc, phi_joint_ref = phi_joint[:n_exc], phi_joint[n_exc:]

        u_joint_eval = propagate_unitary_batched(eval_batch, h_x, h_y, jnp.full(n_ref, w1_hz), jnp.asarray(phi_joint_ref), dt_ref)
        psi0_joint = excite_all(phi_joint_exc)
        snr_joint = float(jnp.abs(_asymptotic_signal(u_joint_eval, psi0_joint, args.n_cycles)))

        result = DurationResult(
            n_ref=n_ref, t_pulse_s=t_pulse, echo_spacing_s=echo_spacing,
            phi_ref=phi_ref, phi_exc_amex=phi_exc_amex, phi_joint=phi_joint,
            snr_rect_rect=snr_rect_rect, snr_grape_ideal=snr_grape_ideal,
            snr_grape_amex=snr_grape_amex, snr_grape_rectexc=snr_grape_rectexc,
            snr_joint=snr_joint,
        )
        duration_results.append(result)

        print(f"-- {n_ref} segments, {t_pulse * 1e6:.1f} us pulse, {echo_spacing * 1e6:.1f} us echo spacing --")
        print(f"   rectangular ref + rectangular exc     : SNR {snr_rect_rect:.4f}  fom_time {result.fom_time(snr_rect_rect):.3e} s")
        print(f"   GRAPE ref (ideal +x exc assumption)    : SNR {snr_grape_ideal:.4f}  fom_time {result.fom_time(snr_grape_ideal):.3e} s")
        print(f"   GRAPE ref + rectangular exc            : SNR {snr_grape_rectexc:.4f}  fom_time {result.fom_time(snr_grape_rectexc):.3e} s")
        print(f"   GRAPE ref + AMEX exc (sequential)      : SNR {snr_grape_amex:.4f}  fom_time {result.fom_time(snr_grape_amex):.3e} s")
        print(f"   joint (ref + exc optimized together)   : SNR {snr_joint:.4f}  fom_time {result.fom_time(snr_joint):.3e} s")

    # Winning combination: best fom_time for the sequential GRAPE+AMEX method.
    best_result = min(duration_results, key=lambda r: r.fom_time(r.snr_grape_amex))
    print()
    print(f"Best SNR/time (GRAPE ref + AMEX exc): {best_result.n_ref} segments "
          f"({best_result.t_pulse_s * 1e6:.1f} us), fom_time = {best_result.fom_time(best_result.snr_grape_amex):.3e} s")
    baseline_fom = best_result.fom_time(best_result.snr_rect_rect)
    print(f"vs. fully rectangular at the same duration: fom_time = {baseline_fom:.3e} s "
          f"({baseline_fom / best_result.fom_time(best_result.snr_grape_amex):.2f}x worse)")

    if args.output is None:
        return

    plt = load_matplotlib(required=True, headless=True)
    n_ref = best_result.n_ref
    dt_ref = jnp.full(n_ref, dt_seg)
    rect_ref = rectangular_seed_phase(n_ref)
    rect_exc = rectangular_seed_phase(n_exc)

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 8))

    # Panel (0,0): refocusing phase waveform.
    ax = axes_grid[0, 0]
    seg_idx = np.arange(n_ref)
    ax.step(seg_idx, np.mod(best_result.phi_ref, 2 * np.pi), where="mid", label="GRAPE-optimized")
    ax.step(seg_idx, np.mod(rect_ref, 2 * np.pi), where="mid", label="rectangular", linestyle="--")
    ax.set_xlabel("refocusing segment index")
    ax.set_ylabel("phase (rad)")
    ax.set_title(f"Refocusing phase ({n_ref} segments, {best_result.t_pulse_s * 1e6:.0f} us)")
    ax.legend(fontsize=8)

    # Panel (0,1): excitation phase waveform.
    ax = axes_grid[0, 1]
    seg_idx_e = np.arange(n_exc)
    ax.step(seg_idx_e, np.mod(best_result.phi_exc_amex, 2 * np.pi), where="mid", label="AMEX-optimized")
    ax.step(seg_idx_e, np.mod(rect_exc, 2 * np.pi), where="mid", label="rectangular", linestyle="--")
    ax.set_xlabel("excitation segment index")
    ax.set_ylabel("phase (rad)")
    ax.set_title("Excitation phase (same peak B1)")
    ax.legend(fontsize=8)

    # Panel (0,2): fom_time across the duration sweep.
    ax = axes_grid[0, 2]
    durations_us = [r.t_pulse_s * 1e6 for r in duration_results]
    ax.plot(durations_us, [r.fom_time(r.snr_rect_rect) for r in duration_results], "o--", label="rect+rect")
    ax.plot(durations_us, [r.fom_time(r.snr_grape_ideal) for r in duration_results], "o--", label="GRAPE ref, ideal exc")
    ax.plot(durations_us, [r.fom_time(r.snr_grape_amex) for r in duration_results], "o-", label="GRAPE ref + AMEX")
    ax.plot(durations_us, [r.fom_time(r.snr_joint) for r in duration_results], "o-", label="joint")
    ax.set_yscale("log")
    ax.set_xlabel("refocusing pulse duration (us)")
    ax.set_ylabel("fom_time = echo_spacing / SNR^2  (s, lower=better)")
    ax.set_title("SNR-per-unit-time trade-off")
    ax.legend(fontsize=7)

    # Panels (1,0)/(1,1): echo-train convergence, GRAPE vs rectangular refocusing.
    u_eval = propagate_unitary_batched(eval_batch, h_x, h_y, jnp.full(n_ref, w1_hz), jnp.asarray(best_result.phi_ref), dt_ref)
    u_rect_eval = propagate_unitary_batched(eval_batch, h_x, h_y, jnp.full(n_ref, w1_hz), jnp.asarray(rect_ref), dt_ref)
    psi_x_eval = jnp.tile(jnp.asarray(PSI_X), (eval_offsets.size, 1))
    axes_grape, _ = jax.vmap(su2_effective_axis)(u_eval)
    targets_grape = jax.vmap(bloch_vector_to_ket)(axes_grape)
    axes_rect, _ = jax.vmap(su2_effective_axis)(u_rect_eval)
    targets_rect = jax.vmap(bloch_vector_to_ket)(axes_rect)

    cycles = np.arange(args.n_cycles + 1)
    for ax, u_batch, targets, title in (
        (axes_grid[1, 0], u_eval, targets_grape, "GRAPE-optimized refocusing"),
        (axes_grid[1, 1], u_rect_eval, targets_rect, "Rectangular refocusing"),
    ):
        traj_x = jax.vmap(iterate_unitary, in_axes=(0, 0, None))(u_batch, psi_x_eval, args.n_cycles)
        sig_x = np.abs(np.asarray(jnp.mean(_transverse(traj_x), axis=0)))
        traj_axis = jax.vmap(iterate_unitary, in_axes=(0, 0, None))(u_batch, targets, args.n_cycles)
        sig_axis = np.abs(np.asarray(jnp.mean(_transverse(traj_axis), axis=0)))
        ax.plot(cycles, sig_x, label="conventional (+x) excitation")
        ax.plot(cycles, sig_axis, label="axis-matched (AMEX) excitation")
        ax.set_xlabel("refocusing cycle index")
        ax.set_ylabel("|ensemble transverse signal|")
        ax.set_title(f"Echo-train transient: {title}")
        ax.legend(fontsize=8)

    # Panel (1,2): asymptotic time-domain echo shape, GRAPE+AMEX vs rectangular+rectangular.
    ax = axes_grid[1, 2]
    wvect = jnp.asarray(TAU * eval_offsets)

    def excite_all_eval(phi_exc, seg_count, dt_arr):
        def one(h_drift_b):
            return propagate_state_scan(h_drift_b, h_x, h_y, jnp.full(seg_count, w1_hz), jnp.asarray(phi_exc), dt_arr, jnp.asarray(PSI_UP))
        return jax.vmap(one)(eval_batch)

    psi0_amex_eval = excite_all_eval(best_result.phi_exc_amex, n_exc, dt_exc)
    traj_amex = jax.vmap(iterate_unitary, in_axes=(0, 0, None))(u_eval, psi0_amex_eval, args.n_cycles)
    mrx_amex = np.asarray(_transverse(traj_amex[:, -1, :]))

    psi0_rect_eval = excite_all_eval(rect_exc, n_exc, dt_exc)
    traj_rect = jax.vmap(iterate_unitary, in_axes=(0, 0, None))(u_rect_eval, psi0_rect_eval, args.n_cycles)
    mrx_rect = np.asarray(_transverse(traj_rect[:, -1, :]))

    # Acquisition window scaled to the offset ensemble's bandwidth (not its
    # point spacing) so the echo's central peak/wings are actually resolved.
    bandwidth_hz = max(eval_offsets[-1] - eval_offsets[0], 1.0)
    tacq = 10.0 / bandwidth_hz
    echo_amex, tvect = calc_time_domain_echo_arb(mrx_amex, np.asarray(wvect), tacq, tacq / 400.0)
    echo_rect, _ = calc_time_domain_echo_arb(mrx_rect, np.asarray(wvect), tacq, tacq / 400.0)
    ax.plot(tvect * 1e6, np.abs(echo_amex), label="GRAPE ref + AMEX exc")
    ax.plot(tvect * 1e6, np.abs(echo_rect), label="rectangular (same peak B1)", linestyle="--")
    ax.set_xlabel("time from echo center (us)")
    ax.set_ylabel("|echo|")
    ax.set_title("Asymptotic echo shape")
    ax.legend(fontsize=8)

    fig.suptitle("GRAPE CPMG refocusing + axis-matched excitation (AMEX)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
