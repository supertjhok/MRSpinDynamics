"""Detector-aware GRAPE: shape the excitation to the readout's noise floor.

A pulse optimized for maximum *detected* SNR should put magnetization where the
detector is quiet. For a broadband SQUID (flat field-noise floor) that means
exciting the whole B0 band; for a band-limited detector (a SERF OPM, whose atomic
response rolls off past ~kHz) it means concentrating the excitation in-band,
because signal outside the band is buried in the detector's rising noise.

This example optimizes an amplitude+phase pulse over an inhomogeneous ``(B0, B1)``
ensemble (single spin-1/2, ``I+`` readout) for two field detectors using
``make_detected_snr_objective`` -- the field-referred detected-SNR objective
(``spin_dynamics.optimal_control.detection_objective``) whose frequency weighting
is the detector's ``1/N^2(f)``:

* a **SQUID** (flat 1 fT/rtHz floor) -> broadband excitation;
* a **SERF OPM** (0.16 fT/rtHz, ~1.5 kHz atomic bandwidth) -> in-band excitation.

A summary (detected SNR before/after, excitation profile widths) prints always;
``--save`` writes a four-panel figure: the two optimized excitation profiles
|M+|(offset), the detectors' noise-weight 1/N^2(f), the optimized pulses, and the
SNR gains.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system  # noqa: E402
from spin_dynamics.detection import OPMMagnetometer, SQUIDMagnetometer  # noqa: E402
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE  # noqa: E402
from spin_dynamics.optimal_control.detection_objective import (  # noqa: E402
    detector_noise_grid,
    make_detected_snr_objective,
)
from spin_dynamics.optimal_control.hamiltonians import coupled_spin_control_model  # noqa: E402
from spin_dynamics.optimal_control.parameterization import amplitude_phase_bounds  # noqa: E402

PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
I_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def _base_operators():
    base = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    return base.h_x, base.h_y


def _drift(offset):
    return coupled_spin_control_model(coupled_spin_system([offset], [[0.0]])).h_drift


def build_ensemble(offsets, b1s):
    hx0, hy0 = _base_operators()
    drift_b, hx_b, hy_b, weights, offs = [], [], [], [], []
    for off in offsets:
        drift = _drift(off)
        for b1 in b1s:
            drift_b.append(drift)
            hx_b.append(b1 * hx0)
            hy_b.append(b1 * hy0)
            weights.append(1.0)
            offs.append(off)
    return drift_b, hx_b, hy_b, np.array(weights), np.array(offs)


def excitation_profile(x, n_segments, dt, offsets_dense):
    """|<I+>|(offset) for the optimized amplitude+phase pulse (B1 = 1, ideal delivery)."""

    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import propagate_batched_controls

    hx0, hy0 = _base_operators()
    amplitude = jnp.asarray(x[:n_segments], dtype=jnp.float64)
    phase = jnp.asarray(x[n_segments : 2 * n_segments], dtype=jnp.float64)
    drift = jnp.stack([jnp.asarray(_drift(o), dtype=jnp.complex128) for o in offsets_dense])
    hx = jnp.broadcast_to(jnp.asarray(hx0, dtype=jnp.complex128), drift.shape)
    hy = jnp.broadcast_to(jnp.asarray(hy0, dtype=jnp.complex128), drift.shape)
    psi0 = jnp.broadcast_to(jnp.asarray(PSI_UP), (drift.shape[0], 2))
    psi = propagate_batched_controls(drift, hx, hy, amplitude, phase, dt, psi0, method="expm")
    detect = jnp.asarray(I_PLUS, dtype=jnp.complex128)
    coh = np.asarray([complex(jnp.vdot(p, detect @ p)) for p in psi])
    return np.abs(coh)


def optimize(vg, x0, bounds, *, starts, seed):
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_x, best_val = x0, vg(x0)[0]
    lower = np.array([lo for lo, _ in bounds])
    upper = np.array([hi for _, hi in bounds])
    for s in range(starts):
        start = x0 if s == 0 else rng.uniform(lower, upper)
        res = minimize(
            lambda x: -vg(x)[0], start, jac=lambda x: -vg(x)[1],
            method="L-BFGS-B", bounds=bounds, options=dict(maxiter=120),
        )
        if -res.fun > best_val:
            best_val, best_x = -res.fun, res.x
    return best_x, best_val


def run(*, n_segments, dt, amp_max, band_hz, opm_bandwidth_hz, starts, seed):
    offsets = np.linspace(-band_hz, band_hz, 15)
    drift_b, hx_b, hy_b, weights, offs = build_ensemble(offsets, [0.85, 1.0, 1.15])
    amp_template = np.full(n_segments, amp_max, dtype=np.float64)
    n_acq, dt_acq = 61, 1.0 / (4.0 * band_hz)  # acquisition band a bit wider than the offsets

    detectors = {
        "SQUID (flat)": SQUIDMagnetometer.berkeley_2007(),
        f"SERF OPM ({opm_bandwidth_hz / 1e3:.1f} kHz)": OPMMagnetometer.serf(
            field_noise_T_per_rtHz=0.16e-15, bandwidth_hz=opm_bandwidth_hz
        ),
    }
    # SNR *per unit RF power* -- the efficiency figure of merit. Under raw SNR,
    # exciting out-of-band never hurts, so every detector wants full excitation;
    # per-power makes wasted (detector-noisy) excitation costly, so the readout
    # shapes the pulse.
    common = dict(
        drift_batch=drift_b, hx_batch=hx_b, hy_batch=hy_b, psi0=PSI_UP,
        detection_operator=I_PLUS, weights=weights, offsets_hz=offs, dt=dt,
        n_segments=n_segments, amplitude_template=amp_template,
        acquisition_points=n_acq, acquisition_dt=dt_acq, optimize_amplitude=True,
        per_rf_power=True,
    )
    bounds = amplitude_phase_bounds(n_segments, amplitude_max_hz=amp_max).as_pairs()
    x0 = np.concatenate([np.full(n_segments, 0.3 * amp_max), np.zeros(n_segments)])
    offsets_dense = np.linspace(-1.5 * band_hz, 1.5 * band_hz, 121)

    results = {}
    for i, (label, det) in enumerate(detectors.items()):
        vg = make_detected_snr_objective(detector=det, **common)
        x_opt, val = optimize(vg, x0, bounds, starts=starts, seed=seed + i)
        results[label] = {
            "detector": det,
            "objective": vg,
            "x": x_opt,
            "snr0": vg(x0)[0],
            "snr": val,
            "profile": excitation_profile(x_opt, n_segments, dt, offsets_dense),
            "noise_grid": detector_noise_grid(det, n_acq, dt_acq),
        }
    # Cross-evaluation: score every optimized pulse under every detector. A truly
    # detector-aware optimum is best under its own readout (diagonal dominance).
    labels = list(results)
    cross = np.array([[results[col]["objective"](results[row]["x"])[0]
                       for col in labels] for row in labels])
    return {
        "results": results,
        "labels": labels,
        "cross": cross,
        "offsets_dense": offsets_dense,
        "n_acq": n_acq,
        "dt_acq": dt_acq,
        "n_segments": n_segments,
        "dt": dt,
        "band_hz": band_hz,
    }


def _profile_width(offsets, profile):
    half = 0.5 * profile.max()
    above = offsets[profile >= half]
    return (above.max() - above.min()) if above.size else 0.0


def print_summary(out) -> None:
    off = out["offsets_dense"]
    print("Detector-aware excitation (single spin-1/2, I+ readout, SNR per RF power)")
    for label, r in out["results"].items():
        print(
            f"  {label:22s}: SNR/pow {r['snr0']:.3e} -> {r['snr']:.3e} "
            f"({r['snr'] / r['snr0']:.2f}x); on-resonance |M+| peak {r['profile'].max():.3f}, "
            f"FWHM ~ {_profile_width(off, r['profile']) / 1e3:.1f} kHz"
        )
    labels = out["labels"]
    cross = out["cross"]
    print("\n  cross-evaluation (SNR/power) -- rows = optimized pulse, cols = detector:")
    print("    " + "".join(f"{c[:16]:>18s}" for c in labels))
    for i, row in enumerate(labels):
        print(f"    {row[:16]:16s}" + "".join(f"{cross[i, j]:18.3e}" for j in range(len(labels))))
    best_own = all(cross[i, i] >= cross[:, i].max() - 1e-30 for i in range(len(labels)))
    print(f"  each pulse best under its own detector: {best_own}")


def make_figure(plt, out):
    off_khz = out["offsets_dense"] / 1e3
    fgrid = np.fft.fftfreq(out["n_acq"], out["dt_acq"]) / 1e3
    order = np.argsort(fgrid)
    results = out["results"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for label, r in results.items():
        ax.plot(off_khz, r["profile"], label=label)
    ax.set_xlabel("B0 offset (kHz)")
    ax.set_ylabel("|M+| (excitation)")
    ax.set_title("(a) Optimized excitation profile")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for label, r in results.items():
        weight = 1.0 / r["noise_grid"]
        ax.plot(fgrid[order], weight[order] / weight.max(), label=label)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("noise weight 1/N^2 (norm.)")
    ax.set_title("(b) Where each detector is quiet")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    n = out["n_segments"]
    t = np.arange(n) * out["dt"] * 1e3
    for label, r in results.items():
        ax.step(t, r["x"][:n], where="mid", label=f"{label} amp")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude (Hz)")
    ax.set_title("(c) Optimized pulse amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    labels = out["labels"]
    cross = out["cross"]
    norm = cross / cross.max(axis=0, keepdims=True)  # normalize per detector (column)
    im = ax.imshow(norm, cmap="viridis", vmin=norm.min(), vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([c.split(" ")[0] for c in labels], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([c.split(" ")[0] for c in labels], fontsize=8)
    ax.set_xlabel("evaluated under detector")
    ax.set_ylabel("pulse optimized for")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.3f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.7 else "black", fontsize=9)
    ax.set_title("(d) Cross-eval SNR/power (per-column norm)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Detector-aware GRAPE: excitation shaped to the readout noise floor")
    fig.tight_layout()
    return fig


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=16, help="pulse segments")
    parser.add_argument("--dt", type=float, default=5e-6, help="segment duration (s)")
    parser.add_argument("--amp-max", type=float, default=4000.0, help="peak B1 (Hz)")
    parser.add_argument("--band-hz", type=float, default=8000.0, help="B0 offset half-band (Hz)")
    parser.add_argument("--opm-bandwidth-hz", type=float, default=600.0, help="SERF OPM bandwidth (Hz)")
    parser.add_argument("--starts", type=int, default=6, help="multistart count")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--save", type=str, default=None, help="path to save the figure")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        raise SystemExit("This example requires the optional 'jax' extra (pip install -e .[jax]).")

    out = run(
        n_segments=args.segments, dt=args.dt, amp_max=args.amp_max, band_hz=args.band_hz,
        opm_bandwidth_hz=args.opm_bandwidth_hz, starts=args.starts, seed=args.seed,
    )
    print_summary(out)

    if args.save:
        plt = load_matplotlib(required=True, headless=True)
        fig = make_figure(plt, out)
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
