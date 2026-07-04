"""Plot the NQR spectrum an EFG-broadened line yields under rectangular vs GRAPE pulses.

An NQR line broadened by an electric-field-gradient (nu_Q) distribution is only
faithfully detected if the excitation pulse creates transverse coherence across
the *whole* line. A fixed-carrier rectangular pulse excites a narrow band and
rolls off at the wings, so the acquired spectrum (single-pulse FID or the first
point of an SLSE echo train) is narrowed and distorted; a GRAPE-optimized
phase-only pulse excites the line uniformly, recovering the true lineshape.

This example (Milestone 3) optimizes a broadband excitation onto the fundamental
transition across a nu_Q ensemble at a fixed carrier (the ``expm`` propagator is
used because quadrupolar spectra are Kramers-degenerate), then, on a fine
held-out grid of spin packets, applies each pulse through the same rotating-frame
NQR physics, forms the excited coherence, and Fourier-transforms a synthesized
FID into a spectrum. It plots (1) the pulse phase waveforms, (2) the excitation
profile |coherence| vs detuning (the pulse's frequency response), and (3) the
resulting NQR spectrum against the true inhomogeneous lineshape.

The inhomogeneity here is the nu_Q / frequency spread (EFG line broadening) --
spin packets, not a powder of orientations -- evaluated at a single RF coupling;
in zero-field NQR the transition frequencies are orientation-independent, so a
powder varies coupling rather than frequency. The broadband-excitation gain shown
here (flattening the rectangular pulse's sinc-like excitation nulls) is a genuine
frequency-domain effect independent of that.

Scope / limitation: this models a **single excitation pulse followed by a free
induction decay only** -- there is no refocusing pulse and no echo train here.
The spectrum shown is the single-pulse (FID) lineshape, equivalently the
first-echo envelope an SLSE train would see under ideal refocusing; it is an
upper bound on, not a simulation of, an SLSE acquisition. For a genuine SLSE
echo train with a GRAPE-optimized offset-robust excitation, see
``plot_grape_nqr_slse_excitation.py``.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

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


def _fundamental_frequency_hz(isotope, nu_q_hz, eta):
    eig = diagonalize_site(quadrupolar_site(isotope, nu_q_hz=nu_q_hz, eta=eta), None)
    return max(eig.transitions, key=lambda t: t.strength).frequency_hz


def _model_for(isotope, nu_q_hz, eta, rf):
    return nqr_site_control_model(
        quadrupolar_site(isotope, nu_q_hz=nu_q_hz, eta=eta), rf_frequency_hz=rf
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotope", type=str, default="63Cu")
    parser.add_argument("--nu-q-mhz", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--nutation-khz", type=float, default=30.0)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--pulse-lengths", type=float, default=2.0,
                        help="Pulse duration in units of 1/(2*nutation).")
    parser.add_argument("--spread-khz", type=float, default=25.0,
                        help="1-sigma of the Gaussian nu_Q (EFG) distribution (kHz).")
    parser.add_argument("--train-points", type=int, default=9)
    parser.add_argument("--grid-points", type=int, default=241,
                        help="Spin packets in the held-out lineshape grid (dense enough "
                        "that the discrete-packet comb merges under the T2* linewidth).")
    parser.add_argument("--t2star-us", type=float, default=200.0)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra. Install it with "
            "`python -m pip install -e .[jax]` (or `.[perf]`)."
        )
        return

    nu_q = args.nu_q_mhz * 1e6
    nutation = args.nutation_khz * 1e3
    sigma = args.spread_khz * 1e3
    n_segments = int(args.segments)
    total = args.pulse_lengths / (2.0 * nutation)
    dt = np.full(n_segments, total / n_segments)

    site0 = quadrupolar_site(args.isotope, nu_q_hz=nu_q, eta=args.eta)
    rf = _default_carrier_hz(diagonalize_site(site0, None))
    model = nqr_site_control_model(site0, rf_frequency_hz=rf)
    lower, upper = nqr_fundamental_states(site0)
    dimension = model.dimension
    psi0 = np.zeros(dimension, dtype=np.complex128)
    psi0[lower] = 1.0
    # Broadband 90: target the equatorial coherence on the fundamental transition.
    target = np.zeros(dimension, dtype=np.complex128)
    target[lower] = 1.0 / np.sqrt(2.0)
    target[upper] = 1j / np.sqrt(2.0)

    # Training ensemble: nu_Q spread (+/- 2 sigma) at the FIXED carrier.
    train_nu_q = nu_q + np.linspace(-2 * sigma, 2 * sigma, int(args.train_points))
    train_batch = [_model_for(args.isotope, nq, args.eta, rf).h_drift for nq in train_nu_q]

    rng = np.random.default_rng(args.seed)
    rect = rectangular_seed_phase(n_segments)
    random_starts = rng.uniform(-np.pi, np.pi, size=(max(args.starts - 1, 1), n_segments))
    initial_phases = np.vstack([rect[np.newaxis, :], random_starts])
    best = run_grape_multistart(
        model, n_segments, initial_phases=initial_phases,
        dt=dt, target=target, psi0=psi0, fixed_amplitude=nutation,
        hamiltonian_batch=train_batch, propagator="expm",
    ).best_result

    # Held-out lineshape grid: Gaussian-weighted spin packets over +/- 3 sigma.
    grid_nu_q = nu_q + np.linspace(-3 * sigma, 3 * sigma, int(args.grid_points))
    weights = np.exp(-0.5 * ((grid_nu_q - nu_q) / sigma) ** 2)
    weights /= weights.sum()
    detuning = np.array(
        [_fundamental_frequency_hz(args.isotope, nq, args.eta) - rf for nq in grid_nu_q]
    )

    def coherence(phase):
        amp = np.full(n_segments, nutation)
        out = np.zeros(grid_nu_q.size, dtype=np.complex128)
        for k, nq in enumerate(grid_nu_q):
            h_drift = _model_for(args.isotope, nq, args.eta, rf).h_drift
            psi = propagate_state_numpy(h_drift, model.h_x, model.h_y, amp, phase, dt, psi0)
            out[k] = psi[lower] * np.conj(psi[upper])  # rho_{lower,upper}
        return out

    c_rect = coherence(rect)
    c_opt = coherence(best.best_phase)

    # Synthesize FIDs and Fourier-transform to spectra. Each spin packet radiates
    # at its rotating-frame detuning, weighted by the lineshape and how well the
    # pulse excited it; an ideal (uniform) excitation gives the true line.
    t2star = args.t2star_us * 1e-6
    n_t = 2048
    t = np.linspace(0.0, 6.0 * t2star, n_t)
    decay = np.exp(-t / t2star)
    phase_mat = np.exp(-1j * 2 * np.pi * np.outer(t, detuning))

    def spectrum(coh):
        fid = (phase_mat * (weights * coh)[None, :]).sum(axis=1) * decay
        return np.abs(np.fft.fftshift(np.fft.fft(fid)))

    freq = np.fft.fftshift(np.fft.fftfreq(n_t, d=t[1] - t[0]))
    spec_true = spectrum(0.5 * np.ones_like(c_rect))  # ideal uniform excitation
    spec_rect = spectrum(c_rect)
    spec_opt = spectrum(c_opt)

    inband = np.abs(grid_nu_q - nu_q) <= 2 * sigma
    print(f"GRAPE broadband NQR excitation spectrum ({args.isotope}, spin {site0.spin:g})")
    print(f"nu_Q: {args.nu_q_mhz:.3f} MHz   eta: {args.eta:.2f}   carrier: {rf / 1e6:.4f} MHz")
    print(f"segments: {n_segments}   nutation: {nutation / 1e3:.0f} kHz   "
          f"duration: {total * 1e6:.1f} us   EFG sigma: {args.spread_khz:.0f} kHz")
    print(f"excitation training fidelity: rectangular {best.initial_fidelity:.4f} "
          f"-> GRAPE {best.best_fidelity:.4f}")
    print(f"in-band coherence |c| (max 0.5): rectangular mean {np.abs(c_rect)[inband].mean():.3f} "
          f"min {np.abs(c_rect)[inband].min():.3f}  ->  "
          f"GRAPE mean {np.abs(c_opt)[inband].mean():.3f} min {np.abs(c_opt)[inband].min():.3f}")

    plt = load_matplotlib(required=True, headless=args.save is not None)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    seg = np.arange(n_segments)
    axes[0].step(seg, np.mod(rect, 2 * np.pi), where="mid", label="rectangular")
    axes[0].step(seg, np.mod(best.best_phase, 2 * np.pi), where="mid", label="GRAPE")
    axes[0].set_xlabel("segment index")
    axes[0].set_ylabel("RF phase (rad)")
    axes[0].set_title("Pulse phase (amplitude fixed)")
    axes[0].legend(fontsize=8)

    det_khz = detuning / 1e3
    axes[1].plot(det_khz, np.abs(c_rect), label="rectangular", lw=1.5)
    axes[1].plot(det_khz, np.abs(c_opt), label="GRAPE", lw=1.5)
    axes[1].axhline(0.5, color="k", lw=0.5, ls=":")
    axes[1].set_xlabel("detuning (kHz)")
    axes[1].set_ylabel(r"$|\rho_{\mathrm{lower,upper}}|$")
    axes[1].set_title("Excitation profile")
    axes[1].legend(fontsize=8)

    fmask = np.abs(freq) <= 4 * sigma
    axes[2].plot(freq[fmask] / 1e3, spec_true[fmask] / spec_true.max(),
                 color="k", ls="--", lw=1.0, label="true line")
    axes[2].plot(freq[fmask] / 1e3, spec_rect[fmask] / spec_true.max(), lw=1.5, label="rectangular")
    axes[2].plot(freq[fmask] / 1e3, spec_opt[fmask] / spec_true.max(), lw=1.5, label="GRAPE")
    axes[2].set_xlabel("frequency offset (kHz)")
    axes[2].set_ylabel("spectral amplitude (norm.)")
    axes[2].set_title("Detected NQR spectrum")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
