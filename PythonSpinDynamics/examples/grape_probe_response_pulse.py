"""GRAPE that accounts for probe bandwidth and gradient-driver dynamics.

Real hardware does not deliver the ideal pulse GRAPE designs. A tuned/matched
RF probe has finite bandwidth (loaded ``Q``) and a ring-down tail, and a
gradient driver has a finite ``L/R`` slew plus eddy-current droop. Each is a
linear time-invariant filter between the *commanded* waveform and the field the
spins see. ``spin_dynamics.optimal_control.control_response`` models them as one
differentiable stage the objective inserts in the loop, so:

* fidelity is scored on the **delivered** (filtered) pulse, including the
  ring-down tail; and
* box constraints stay on the **command**, so the optimizer pre-emphasizes the
  command only up to the real hardware limits.

Part 1 optimizes a phase-only inversion pulse *through* a band-limited tuned
probe and shows it recovers the fidelity a rectangular pulse loses to the probe.
Part 2 illustrates the new gradient-driver model (RL slew + eddy shelves) and
the step-response eddy-fit helper.
"""

from __future__ import annotations

import argparse
from math import ceil

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.control_response import (
    GradientDriverResponse,
    TunedProbeResponse,
    eddy_terms_from_step_response,
)
from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model
from spin_dynamics.optimal_control.objectives import make_grape_objective
from spin_dynamics.optimal_control.parameterization import rectangular_seed_phase
from spin_dynamics.optimal_control.solvers import grape_optimize

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_PSI_DOWN = np.array([0.0, 1.0], dtype=np.complex128)


def _model(offset_hz: float):
    return coupled_spin_control_model(coupled_spin_system([offset_hz], [[0.0]]))


def _delivered_envelope(response, amplitude_hz, phase, dt):
    """Reconstruct the delivered complex envelope on the fine grid (for plotting).

    Mirrors the objective's transform: upsample the piecewise-constant command,
    append the ring-down tail, and filter -- host-side NumPy, no autodiff.
    """

    s = int(response.oversample)
    u = amplitude_hz * np.exp(1j * np.asarray(phase))
    dt_fine = dt / s
    n_tail = ceil(response.tail.seconds(response.dominant_tau) / dt_fine)
    u_fine = np.concatenate([np.repeat(u, s), np.zeros(n_tail, dtype=complex)])
    return response.apply(u_fine, dt_fine, xp=np), dt_fine


def _probe_part(args) -> None:
    n = int(args.segments)
    dt = float(args.pulse_duration_us) * 1e-6 / n
    nutation_hz = float(args.nutation_hz)
    model = _model(float(args.offset_hz))

    probe = TunedProbeResponse.from_quality_factor(
        f0_hz=float(args.probe_f0_mhz) * 1e6,
        quality_factor=float(args.probe_q),
        oversample=int(args.oversample),
    )
    tau_us = probe.tau * 1e6

    common = dict(dt=dt, target=_PSI_DOWN, psi0=_PSI_UP, fixed_amplitude=nutation_hz)

    # Rectangular baseline, scored on the delivered (filtered) pulse.
    rect = rectangular_seed_phase(n)
    vg_probe = make_grape_objective(model, n_segments=n, rf_response=probe, **common)
    rect_fid = float(vg_probe(rect)[0])

    # GRAPE optimizing the command against the delivered pulse.
    rng = np.random.default_rng(int(args.seed))
    best = None
    for trial in range(int(args.starts)):
        seed = rect if trial == 0 else rng.uniform(-np.pi, np.pi, size=n)
        result = grape_optimize(
            model, seed, rf_response=probe,
            scipy_options={"maxiter": int(args.maxiter)}, **common,
        )
        if best is None or result.best_fidelity > best.best_fidelity:
            best = result

    # Reference: the same optimization with an ideal (unfiltered) probe -- how
    # much of the gap is the probe versus the pulse design itself.
    ideal = grape_optimize(
        model, rect, scipy_options={"maxiter": int(args.maxiter)}, **common
    )

    print("Part 1 -- phase-only inversion through a band-limited tuned probe")
    print(f"  probe: f0 = {args.probe_f0_mhz:.3f} MHz, Q = {args.probe_q:.0f} "
          f"-> ring-down tau = {tau_us:.2f} us")
    print(f"  pulse: {n} segments, {dt * 1e6:.2f} us each, nutation {nutation_hz:.0f} Hz, "
          f"offset {args.offset_hz:.0f} Hz")
    print(f"  rectangular (delivered)      : {rect_fid:.4f}")
    print(f"  GRAPE through probe (delivered): {best.best_fidelity:.4f}  "
          f"[{best.iterations} iters]")
    print(f"  GRAPE with ideal probe (ref)  : {ideal.best_fidelity:.4f}")
    print(f"  recovery: {best.best_fidelity - rect_fid:+.4f} vs rectangular")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        delivered, dt_fine = _delivered_envelope(probe, nutation_hz, best.best_phase, dt)
        t_fine = np.arange(delivered.size) * dt_fine * 1e6
        cmd_t = (np.arange(n) + 0.5) * dt * 1e6
        axes[0].plot(t_fine, np.abs(delivered), lw=1.2, label="delivered |B1|")
        axes[0].axhline(nutation_hz, ls="--", color="gray", lw=1, label="commanded |B1|")
        axes[0].set_xlabel("time (us)")
        axes[0].set_ylabel("nutation rate (Hz)")
        axes[0].set_title("Delivered envelope (roll-off + ring-down tail)")
        axes[0].legend()
        axes[1].step(cmd_t, np.mod(best.best_phase, 2 * np.pi), where="mid")
        axes[1].set_xlabel("time (us)")
        axes[1].set_ylabel("commanded phase (rad)")
        axes[1].set_title("Optimized command (pre-emphasized for the probe)")
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


def _gradient_part() -> None:
    print("\nPart 2 -- gradient driver (RL slew + eddy currents)")
    dt = 1e-6
    driver = GradientDriverResponse(tau_rl=2e-6, eddy_terms=((0.3, 5e-5), (0.1, 3e-4)))
    delivered = driver.apply(np.ones(1000), dt, xp=np)
    print(f"  driver: tau_RL = {driver.tau_rl * 1e6:.1f} us, "
          f"eddy terms = {[(a, f'{tau * 1e6:.0f}us') for a, tau in driver.eddy_terms]}")
    print(f"  unit step response: initial {delivered[1]:.3f}, "
          f"at 50 us {delivered[50]:.3f}, settled {delivered[-1]:.3f} (DC gain 1)")

    # The eddy-fit helper recovers (alpha_k, tau_k) from a measured step response
    # of the classic eddy form G_inf*(1 - sum_k alpha_k*exp(-t/tau_k)). Round-trip
    # it here against a two-term response built from known terms.
    true_terms = ((0.25, 4e-5), (0.08, 2.5e-4))
    t = np.arange(0.0, 2e-3, 1e-6)
    measured = 1.0 * (1.0 - sum(a * np.exp(-t / tau) for a, tau in true_terms))
    fitted = eddy_terms_from_step_response(t, measured, n_terms=2)
    print("  eddy-fit helper (true -> recovered):")
    for (a_t, tau_t), (a_f, tau_f) in zip(true_terms, fitted):
        print(f"    alpha {a_t:.3f} -> {a_f:.3f}   tau {tau_t * 1e6:.0f} us -> {tau_f * 1e6:.0f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--nutation-hz", type=float, default=6000.0)
    parser.add_argument("--pulse-duration-us", type=float, default=240.0)
    parser.add_argument("--offset-hz", type=float, default=1500.0)
    parser.add_argument("--probe-f0-mhz", type=float, default=1.0)
    parser.add_argument("--probe-q", type=float, default=120.0)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="Optional summary figure path.")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra. Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    _probe_part(args)
    _gradient_part()


if __name__ == "__main__":
    main()
