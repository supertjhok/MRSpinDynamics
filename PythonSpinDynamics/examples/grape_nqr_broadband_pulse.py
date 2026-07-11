"""GRAPE a broadband NQR inversion pulse robust to EFG inhomogeneity.

Real NQR lines (energetic-materials detection, ¹⁴N/³⁵Cl/⁶³Cu, ...) are
inhomogeneously broadened: a distribution of electric-field-gradient magnitudes
spreads the quadrupole frequency nu_Q by kHz to tens of kHz, so a fixed-carrier
rectangular pulse only inverts the spin packets near resonance and fails at the
band edges. GRAPE optimizes the per-segment RF phase (amplitude held constant --
the primary mode for switching-power-amplifier field hardware) to invert the
fundamental transition robustly across the whole broadened line. (The ensemble
here is the nu_Q / frequency spread -- EFG line broadening -- not a powder of
orientations; in zero-field NQR the transition frequencies are
orientation-independent, so orientation varies the RF coupling, not the
frequency. See ``grape_nqr_powder_pulse.py`` for the orientation axis.)

This runs the ``spin_dynamics.optimal_control`` engine on a quadrupolar target
(Milestone 3): the model is built in the rotating frame the NQR density-matrix
engine uses (``nqr.full_dynamics``), so the optimized pulse round-trips through
the real simulator. Because quadrupolar spectra are Kramers-degenerate, the
degeneracy-safe ``expm`` propagator is used (the default ``eigh`` gradient is
NaN under exact degeneracy). The ensemble is a grid of nu_Q values at a FIXED
carrier -- the physically-meaningful detuning distribution. The rectangular
baseline is a constant-phase pulse of the *same duration and nutation* as the
GRAPE pulse (the same time budget), so the comparison isolates what pulse
shaping buys; both are checked on a finer held-out nu_Q grid.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.nqr.full_dynamics import _default_carrier_hz
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.isotopes import quadrupolar_site
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_state_numpy,
)
from spin_dynamics.optimal_control.drivers import run_grape_multistart
from spin_dynamics.optimal_control.hamiltonians import (
    nqr_fundamental_states,
    nqr_site_control_model,
)
from spin_dynamics.optimal_control.parameterization import rectangular_seed_phase


def _drift_for_nu_q(isotope, nu_q_hz, eta, rf_hz):
    """h_drift for a detuned spin packet: same fixed carrier, shifted nu_Q."""

    site = quadrupolar_site(isotope, nu_q_hz=nu_q_hz, eta=eta)
    return nqr_site_control_model(site, rf_frequency_hz=rf_hz).h_drift


def _inversion_fidelity(h_drift_list, h_x, h_y, phase, nutation_hz, dt, psi0, target):
    fids = np.zeros(len(h_drift_list))
    amp = np.full(phase.size, nutation_hz)
    for idx, h_drift in enumerate(h_drift_list):
        psi = propagate_state_numpy(h_drift, h_x, h_y, amp, phase, dt, psi0)
        overlap = np.vdot(target, psi)
        fids[idx] = np.real(overlap * np.conj(overlap))
    return fids


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotope", type=str, default="63Cu", help="Quadrupolar isotope.")
    parser.add_argument("--nu-q-mhz", type=float, default=2.0, help="Nominal nu_Q (MHz).")
    parser.add_argument("--eta", type=float, default=0.1, help="EFG asymmetry parameter.")
    parser.add_argument("--nutation-khz", type=float, default=30.0, help="Fixed RF nutation (kHz).")
    parser.add_argument("--segments", type=int, default=12, help="Number of phase segments.")
    parser.add_argument("--pulse-lengths", type=float, default=3.0,
                        help="Pulse duration in units of 1/(2*nutation) (inversion times).")
    parser.add_argument("--spread-khz", type=float, default=25.0, help="Half-width of the nu_Q spread (kHz).")
    parser.add_argument("--train-points", type=int, default=7, help="nu_Q values in the training ensemble.")
    parser.add_argument("--eval-points", type=int, default=41, help="nu_Q values in the held-out grid.")
    parser.add_argument("--starts", type=int, default=12, help="Random-phase restarts.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save a summary figure.")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra (reverse-mode autodiff is the "
            "whole point of the engine). Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    nu_q = args.nu_q_mhz * 1e6
    nutation = args.nutation_khz * 1e3
    spread = args.spread_khz * 1e3
    n_segments = int(args.segments)
    total_duration = args.pulse_lengths / (2.0 * nutation)
    dt = np.full(n_segments, total_duration / n_segments)

    site0 = quadrupolar_site(args.isotope, nu_q_hz=nu_q, eta=args.eta)
    # One FIXED carrier (the nominal fundamental line) shared across the ensemble.
    rf = _default_carrier_hz(diagonalize_site(site0, None))
    model = nqr_site_control_model(site0, rf_frequency_hz=rf)
    lower, upper = nqr_fundamental_states(site0)
    dimension = model.dimension
    psi0 = np.zeros(dimension, dtype=np.complex128)
    psi0[lower] = 1.0
    target = np.zeros(dimension, dtype=np.complex128)
    target[upper] = 1.0

    train_nu_q = nu_q + np.linspace(-spread, spread, int(args.train_points))
    train_batch = [_drift_for_nu_q(args.isotope, nq, args.eta, rf) for nq in train_nu_q]

    rng = np.random.default_rng(args.seed)
    rect = rectangular_seed_phase(n_segments)
    random_starts = rng.uniform(-np.pi, np.pi, size=(max(args.starts - 1, 1), n_segments))
    initial_phases = np.vstack([rect[np.newaxis, :], random_starts])

    multistart = run_grape_multistart(
        model, n_segments,
        initial_phases=initial_phases,
        dt=dt, target=target, psi0=psi0,
        fixed_amplitude=nutation,
        hamiltonian_batch=train_batch,
        propagator="expm",
    )
    best = multistart.best_result

    eval_nu_q = nu_q + np.linspace(-spread, spread, int(args.eval_points))
    eval_batch = [_drift_for_nu_q(args.isotope, nq, args.eta, rf) for nq in eval_nu_q]
    rect_fids = _inversion_fidelity(eval_batch, model.h_x, model.h_y, rect, nutation, dt, psi0, target)
    opt_fids = _inversion_fidelity(eval_batch, model.h_x, model.h_y, best.best_phase, nutation, dt, psi0, target)

    print(f"GRAPE broadband NQR inversion ({args.isotope}, spin {site0.spin:g})")
    print(f"nu_Q: {args.nu_q_mhz:.3f} MHz   eta: {args.eta:.2f}   carrier: {rf / 1e6:.4f} MHz")
    print(f"segments: {n_segments}   nutation: {nutation / 1e3:.0f} kHz (fixed)   "
          f"duration: {total_duration * 1e6:.1f} us")
    print(f"EFG spread: +/- {args.spread_khz:.0f} kHz nu_Q   "
          f"training points: {args.train_points}   held-out: {args.eval_points}   restarts: {args.starts}")
    print(f"fundamental transition eigenstates: |{lower}> -> |{upper}>   "
          f"optimizer iterations: {best.iterations}")
    print()
    print(f"training-ensemble mean fidelity: rectangular {best.initial_fidelity:.4f} "
          f"-> optimized {best.best_fidelity:.4f}")
    print(f"held-out mean inversion : rectangular {rect_fids.mean():.4f} -> optimized {opt_fids.mean():.4f}")
    print(f"held-out worst-case     : rectangular {rect_fids.min():.4f} -> optimized {opt_fids.min():.4f}")

    if args.save is not None:
        from _source_path import load_matplotlib

        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        detuning_khz = (eval_nu_q - nu_q) / 1e3
        axes[0].plot(detuning_khz, rect_fids, label="rectangular", lw=1.5)
        axes[0].plot(detuning_khz, opt_fids, label="GRAPE-optimized", lw=1.5)
        axes[0].axvspan(-args.spread_khz, args.spread_khz, color="gray", alpha=0.08)
        axes[0].set_xlabel(r"$\nu_Q$ detuning (kHz)")
        axes[0].set_ylabel("inversion fidelity")
        axes[0].set_title("Held-out inversion vs. EFG detuning")
        axes[0].legend()

        seg = np.arange(n_segments)
        axes[1].step(seg, np.mod(best.best_phase, 2 * np.pi), where="mid")
        axes[1].set_xlabel("segment index")
        axes[1].set_ylabel("phase (rad)")
        axes[1].set_title("Optimized phase (amplitude fixed)")

        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
