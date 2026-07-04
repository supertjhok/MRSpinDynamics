"""GRAPE an orientation-robust (powder) NQR inversion pulse.

An NQR sample is a powder: every crystallite presents a different orientation of
its electric-field-gradient principal-axis system to the RF coil, so the RF-to-
spin coupling -- and hence the effective nutation about the fundamental
transition -- varies across the powder. A single fixed-duration rectangular
pulse therefore over-rotates some crystallites and under-rotates others. GRAPE
optimizes the per-segment RF phase (amplitude fixed) to invert the fundamental
transition robustly across all orientations at once.

This is the powder counterpart to ``grape_nqr_broadband_pulse.py`` and exercises
the per-case control-operator path of the engine (Milestone 3): the drive
operators ``h_x``/``h_y`` vary per crystallite (``nqr_powder_control_batch``)
while a single shared RF waveform is optimized. The degeneracy-safe ``expm``
propagator is used (quadrupolar spectra are Kramers-degenerate). Trained on a
coarse powder grid, the pulse is checked on a finer held-out grid.

Scope: this is a *single-pulse* inversion. The powder RF-coupling distribution is
mild for spin-3/2 (no orientations with zero coupling), so a single robust
inversion is achievable. This is a different, easier problem than powder-averaged
*detection trains* (SLSE/SORC), where the refocusing cycle amplifies
orientation differences into near-nulls and a well-tuned magic-angle rectangular
sequence is already hard to beat -- see ``plot_grape_nqr_slse_excitation.py`` and
the manual for that distinction.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.nqr.full_dynamics import _default_carrier_hz
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.isotopes import quadrupolar_site
from spin_dynamics.nqr.orientations import powder_average_grid
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_state_numpy,
)
from spin_dynamics.optimal_control.drivers import run_grape_multistart
from spin_dynamics.optimal_control.hamiltonians import (
    nqr_fundamental_states,
    nqr_powder_control_batch,
    nqr_site_control_model,
)
from spin_dynamics.optimal_control.parameterization import rectangular_seed_phase


def _powder_inversion_fidelity(control_batch, h_drift, phase, nutation_hz, dt, psi0, target):
    fids = np.zeros(len(control_batch))
    amp = np.full(phase.size, nutation_hz)
    for idx, (h_x, h_y) in enumerate(control_batch):
        psi = propagate_state_numpy(h_drift, h_x, h_y, amp, phase, dt, psi0)
        overlap = np.vdot(target, psi)
        fids[idx] = np.real(overlap * np.conj(overlap))
    return fids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotope", type=str, default="35Cl", help="Quadrupolar isotope.")
    parser.add_argument("--nu-q-mhz", type=float, default=1.5, help="nu_Q (MHz).")
    parser.add_argument("--eta", type=float, default=0.0, help="EFG asymmetry parameter.")
    parser.add_argument("--nutation-khz", type=float, default=30.0, help="Fixed RF nutation (kHz).")
    parser.add_argument("--segments", type=int, default=12, help="Number of phase segments.")
    parser.add_argument("--pulse-lengths", type=float, default=3.0,
                        help="Pulse duration in units of 1/(2*nutation).")
    parser.add_argument("--train-theta", type=int, default=5, help="Polar nodes in the training powder grid.")
    parser.add_argument("--train-phi", type=int, default=8, help="Azimuthal nodes in the training powder grid.")
    parser.add_argument("--eval-theta", type=int, default=8, help="Polar nodes in the held-out grid.")
    parser.add_argument("--eval-phi", type=int, default=12, help="Azimuthal nodes in the held-out grid.")
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
    n_segments = int(args.segments)
    total_duration = args.pulse_lengths / (2.0 * nutation)
    dt = np.full(n_segments, total_duration / n_segments)

    site = quadrupolar_site(args.isotope, nu_q_hz=nu_q, eta=args.eta)
    rf = _default_carrier_hz(diagonalize_site(site, None))
    model = nqr_site_control_model(site, rf_frequency_hz=rf)
    lower, upper = nqr_fundamental_states(site)
    dimension = model.dimension
    psi0 = np.zeros(dimension, dtype=np.complex128)
    psi0[lower] = 1.0
    target = np.zeros(dimension, dtype=np.complex128)
    target[upper] = 1.0

    train_orient = powder_average_grid(n_theta=int(args.train_theta), n_phi=int(args.train_phi))
    train_batch = nqr_powder_control_batch(site, train_orient, rf_frequency_hz=rf)

    rng = np.random.default_rng(args.seed)
    rect = rectangular_seed_phase(n_segments)
    random_starts = rng.uniform(-np.pi, np.pi, size=(max(args.starts - 1, 1), n_segments))
    initial_phases = np.vstack([rect[np.newaxis, :], random_starts])

    multistart = run_grape_multistart(
        model, n_segments,
        initial_phases=initial_phases,
        dt=dt, target=target, psi0=psi0,
        fixed_amplitude=nutation,
        control_operator_batch=train_batch,
        propagator="expm",
    )
    best = multistart.best_result

    eval_orient = powder_average_grid(n_theta=int(args.eval_theta), n_phi=int(args.eval_phi))
    eval_batch = nqr_powder_control_batch(site, eval_orient, rf_frequency_hz=rf)
    rect_fids = _powder_inversion_fidelity(eval_batch, model.h_drift, rect, nutation, dt, psi0, target)
    opt_fids = _powder_inversion_fidelity(eval_batch, model.h_drift, best.best_phase, nutation, dt, psi0, target)

    print(f"GRAPE powder-robust NQR inversion ({args.isotope}, spin {site.spin:g})")
    print(f"nu_Q: {args.nu_q_mhz:.3f} MHz   eta: {args.eta:.2f}   carrier: {rf / 1e6:.4f} MHz")
    print(f"segments: {n_segments}   nutation: {nutation / 1e3:.0f} kHz (fixed)   "
          f"duration: {total_duration * 1e6:.1f} us")
    print(f"training grid: {args.train_theta}x{args.train_phi} = {len(train_batch)} orientations   "
          f"held-out: {args.eval_theta}x{args.eval_phi} = {len(eval_batch)}   restarts: {args.starts}")
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
        order = np.argsort(rect_fids)
        axes[0].plot(np.sort(rect_fids), label="rectangular", lw=1.5)
        axes[0].plot(opt_fids[order], label="GRAPE-optimized", lw=1.5)
        axes[0].set_xlabel("crystallite (sorted by rectangular fidelity)")
        axes[0].set_ylabel("inversion fidelity")
        axes[0].set_title("Held-out inversion across the powder")
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
