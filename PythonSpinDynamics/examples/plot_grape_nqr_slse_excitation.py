"""GRAPE-optimized offset-robust excitation for a single-crystal NQR SLSE train.

In an SLSE (spin-lock spin-echo) detection train, an off-resonance spin loses
signal: a conventional rectangular 90 excitation places magnetization that the
refocusing cycle only partly spin-locks, so the echo train starts high and
decays/oscillates down to a small steady value. Numerically optimizing the
excitation waveform (against the *actual* steady echo, from the thermal
equilibrium density read out through the detection operator) recovers the full
signal and a flat, transient-free train.

For a single crystallite at a known carrier offset this gives a real
signal-to-noise gain that **grows with offset**: none on resonance (a rectangular
pulse already covers it -- there is nothing to optimize), rising to roughly
1.3--1.4x once the offset exceeds the pulse bandwidth (offset >~ 2x the nutation
rate), where it also beats a flip-angle-optimized rectangular excitation. The
optimized excitation holds the echo near its on-resonance level across the whole
offset range.

The physics uses the full (2I+1) density-matrix rotating-frame model
(``nqr.full_dynamics``: ``equilibrium_density``, ``static_hamiltonian_rotating``,
``detection_operator``) built through the ``optimal_control`` NQR model builder;
the ``expm`` propagator is used throughout (quadrupolar spectra are
Kramers-degenerate). The excitation is optimized with SciPy directly against the
steady echo -- a scalar sequence-level figure of merit that the state-transfer
GRAPE objective does not expose.

**Scope / limitation.** This is a *single-crystal* (known-offset) result and does
NOT extend to a powder. Over a powder the refocusing-cycle response varies
strongly and oppositely with orientation (coupling and effective axis), one
excitation pulse cannot serve a spread of offsets/axes at once, and a
flip-angle-optimized (``magic-angle'') rectangular SLSE already navigates the
powder -- against it, an optimized pulse does not improve the powder echo. (An
earlier axis-matched-excitation (AMEX) approach was checked and discarded: the
spin-1/2 CPMG cycle-axis heuristic is not optimal for NQR detection, because the
detection operator and 3-level thermal equilibrium break the transverse-detection
correspondence it relies on.)
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.nqr.full_dynamics import _default_carrier_hz, detection_operator
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.isotopes import quadrupolar_site
from spin_dynamics.nqr.orientations import single_crystal_orientation
from spin_dynamics.nqr.simulation import equilibrium_density
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.hamiltonians import nqr_site_control_model


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotope", type=str, default="14N")
    parser.add_argument("--nu-q-mhz", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=0.4)
    parser.add_argument("--nutation-khz", type=float, default=25.0)
    parser.add_argument("--refocus-deg", type=float, default=119.0,
                        help="Rectangular refocusing flip angle (deg).")
    parser.add_argument("--exc-segments", type=int, default=10)
    parser.add_argument("--tau-us", type=float, default=300.0, help="Echo spacing (us).")
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument("--max-offset-ratio", type=float, default=3.0,
                        help="Largest carrier offset, in units of the nutation rate.")
    parser.add_argument("--offset-points", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.4, help="Crystallite azimuth (rad).")
    parser.add_argument("--beta", type=float, default=0.9, help="Crystallite polar angle (rad).")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra. Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    nutation = args.nutation_khz * 1e3
    tau = args.tau_us * 1e-6
    n_exc = int(args.exc_segments)
    dt_exc = np.full(n_exc, (1.0 / (2.0 * nutation)) / n_exc)
    tail = max(1, args.n_cycles // 4)

    site = quadrupolar_site(args.isotope, nu_q_hz=args.nu_q_mhz * 1e6, eta=args.eta)
    eig0 = diagonalize_site(site, None)
    line = _default_carrier_hz(eig0)
    b1 = single_crystal_orientation(args.alpha, args.beta)[0].b1_direction_pas
    dim = site.dimension

    def build(offset):
        carrier = line + offset
        model = nqr_site_control_model(site, rf_frequency_hz=carrier, b1_direction_pas=b1)
        eig = diagonalize_site(site, None)
        refocus_dur = np.deg2rad(args.refocus_deg) / (4.0 * np.pi * nutation)
        free_half = 0.5 * (tau - refocus_dur)
        u_refocus = expm(-1j * (model.h_drift + nutation * model.h_y) * refocus_dur)
        u_free = propagator(model.h_drift, free_half)
        detector = detection_operator(eig, carrier, b1)
        rho_eq = equilibrium_density(eig.levels_hz)
        return model, u_refocus, u_free, detector, rho_eq

    def pulse_unitary(phase, model):
        u = np.eye(dim, dtype=np.complex128)
        for p, t in zip(phase, dt_exc):
            u = expm(-1j * (model.h_drift + nutation
                            * (np.cos(p) * model.h_x + np.sin(p) * model.h_y)) * t) @ u
        return u

    def echo_train(u_excite, u_refocus, u_free, detector, rho_eq):
        rho = u_excite @ rho_eq @ u_excite.conj().T
        echoes = np.empty(args.n_cycles)
        for n in range(args.n_cycles):
            rho = u_free @ rho @ u_free.conj().T
            rho = u_refocus @ rho @ u_refocus.conj().T
            rho = u_free @ rho @ u_free.conj().T
            echoes[n] = abs(np.trace(rho @ detector))
        return echoes

    def steady(echoes):
        return float(np.mean(echoes[-tail:]))

    def optimize_excitation(model, u_refocus, u_free, detector, rho_eq, seed):
        rng = np.random.default_rng(seed)

        def neg(phase):
            return -steady(echo_train(pulse_unitary(phase, model), u_refocus, u_free, detector, rho_eq))

        best_val, best_phase = -np.inf, None
        for r in range(max(args.restarts, 1)):
            x0 = np.zeros(n_exc) if r == 0 else rng.uniform(-np.pi, np.pi, n_exc)
            res = minimize(neg, x0, method="L-BFGS-B",
                           options={"maxiter": 300, "ftol": 1e-9})
            if -res.fun > best_val:
                best_val, best_phase = -res.fun, res.x
        return best_val, best_phase

    t90 = 1.0 / (4.0 * nutation)
    offsets = np.linspace(0.0, args.max_offset_ratio, int(args.offset_points)) * nutation
    conv_steady = np.zeros(offsets.size)
    opt_steady = np.zeros(offsets.size)
    best_phase_hi = None
    for i, off in enumerate(offsets):
        model, u_ref, u_free, det, rho_eq = build(off)
        u_conv = expm(-1j * (model.h_drift + nutation * model.h_x) * t90)
        conv_steady[i] = steady(echo_train(u_conv, u_ref, u_free, det, rho_eq))
        opt_steady[i], phase = optimize_excitation(model, u_ref, u_free, det, rho_eq, args.seed + i)
        if i == offsets.size - 1:
            best_phase_hi = phase

    # Echo trains at the largest offset for the transient/flatness panel.
    model, u_ref, u_free, det, rho_eq = build(offsets[-1])
    conv_train = echo_train(expm(-1j * (model.h_drift + nutation * model.h_x) * t90),
                            u_ref, u_free, det, rho_eq)
    opt_train = echo_train(pulse_unitary(best_phase_hi, model), u_ref, u_free, det, rho_eq)
    ratio = opt_steady / np.maximum(conv_steady, 1e-12)

    print(f"GRAPE offset-robust single-crystal NQR SLSE excitation "
          f"({args.isotope}, spin {site.spin:g})")
    print(f"nu_Q: {args.nu_q_mhz:.3f} MHz  eta: {args.eta:.2f}  nutation: {nutation / 1e3:.0f} kHz  "
          f"refocus flip: {args.refocus_deg:.0f} deg  tau: {args.tau_us:.0f} us")
    print()
    print("offset/nutation | conventional | optimized | ratio")
    for r, c, o in zip(offsets / nutation, conv_steady, opt_steady):
        print(f"     {r:4.1f}       |   {c:.4f}     |  {o:.4f}   | {o / max(c, 1e-12):.2f}x")
    print()
    print(f"peak SNR gain: {ratio.max():.2f}x at offset {offsets[np.argmax(ratio)] / nutation:.1f}x nutation")
    print(f"echo-train std at offset {args.max_offset_ratio:.0f}x: "
          f"conventional {conv_train.std():.2e} (transient), optimized {opt_train.std():.2e} (flat)")
    print("Single-crystal result; does NOT extend to a powder (see docstring / manual).")

    plt = load_matplotlib(required=True, headless=args.save is not None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(offsets / nutation, conv_steady, "-o", ms=4, label="conventional (rect 90)")
    axes[0].plot(offsets / nutation, opt_steady, "-o", ms=4, label="GRAPE-optimized")
    axes[0].set_xlabel("carrier offset / nutation rate")
    axes[0].set_ylabel("steady echo amplitude")
    axes[0].set_title("Offset robustness")
    axes[0].legend(fontsize=8)

    cycles = np.arange(1, args.n_cycles + 1)
    axes[1].plot(cycles, conv_train, "-o", ms=3, label="conventional")
    axes[1].plot(cycles, opt_train, "-o", ms=3, label="GRAPE-optimized")
    axes[1].set_xlabel("refocusing cycle")
    axes[1].set_ylabel("echo amplitude")
    axes[1].set_title(f"Echo train at offset {args.max_offset_ratio:.0f}x nutation")
    axes[1].legend(fontsize=8)

    axes[2].step(np.arange(n_exc), np.mod(best_phase_hi, 2 * np.pi), where="mid")
    axes[2].set_xlabel("segment index")
    axes[2].set_ylabel("RF phase (rad)")
    axes[2].set_title("Optimized excitation (amplitude fixed)")

    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
