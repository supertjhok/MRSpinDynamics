r"""Balanced SSFP imaging in mildly inhomogeneous (B0, B1) fields.

Balanced SSFP (TrueFISP / FIESTA) is the high-SNR steady-state gradient echo: on
resonance its signal is ``~ M0/2 * sqrt(T2/T1)`` at the optimal flip, which is why
it is attractive at low field where SNR is scarce. The catch is the flip side of
"balanced": each ``TR`` rewinds the *gradient* phase but not the *B0* phase, so
the steady state depends on the per-``TR`` off-resonance precession and its
off-resonance response has periodic nulls -- the bSSFP "dark bands" -- spaced
``1/TR`` in frequency. In an inhomogeneous B0 those bands cut across the object as
signal voids, and B1 (flip-angle) inhomogeneity shades and reshapes the response.
This is the steady-state SNR-versus-robustness trade-off, the MRI cousin of the
steady-state SORC story in the NQR examples.

The sequence (``run_bssfp_imaging``) runs on the same moving-isochromat engine and
the same B0/B1 field maps as the spin-echo imagers, so this is an apples-to-apples
setup: a continuous steady-state train of balanced TRs with a phase-cycled RF
phase, one k-space line per TR.

Panels:

1. Off-resonance response -- the banding profile, for RF phase increments of 180
   (passband centred on resonance, nulls at +/- 1/2TR) and 0 (nulls at 0, +/-
   1/TR). Band spacing is 1/TR.
2. Signal vs flip angle at band centre -- the T2/T1 contrast and the optimal flip
   ``arccos((T1/T2 - 1)/(T1/T2 + 1))``, peaking at ``M0/2 sqrt(T2/T1)``.
3. Reference image -- uniform fields.
4. Mild B0 inhomogeneity -> banding (signal voids where the local offset hits a
   null).
5. Mild transmit-B1 inhomogeneity -> intensity shading.
6. Phase-cycled combination -- the same B0 case imaged at RF increments 180 and 0
   and combined (maximum), the standard banding fix, at twice the scan time.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


# Keep CLI choices together so scientific defaults are easy to find and override.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Balanced SSFP imaging in mildly inhomogeneous B0/B1 fields: the "
            "off-resonance banding profile, the T2/T1 flip-angle response, and "
            "images showing banding, B1 shading, and phase-cycled band removal."
        )
    )
    parser.add_argument("--pixels", type=int, default=32, help="Image size.")
    parser.add_argument("--flip-deg", type=float, default=50.0, help="Nominal flip angle (deg).")
    parser.add_argument("--readout-time-ms", type=float, default=1.6,
                        help="Readout duration (ms); sets TR and so the 1/TR band spacing.")
    parser.add_argument("--t1-ms", type=float, default=300.0, help="Phantom T1 (ms).")
    parser.add_argument("--t2-ms", type=float, default=100.0, help="Phantom T2 (ms).")
    parser.add_argument("--b0-inhomogeneity-hz", type=float, default=430.0,
                        help="Peak B0 off-resonance across the FOV (Hz).")
    parser.add_argument("--b1-min", type=float, default=0.65,
                        help="Minimum transmit-B1 fraction across the FOV.")
    parser.add_argument("--b1-max", type=float, default=1.2,
                        help="Maximum transmit-B1 fraction across the FOV.")
    parser.add_argument("--num-dummy-tr", type=int, default=200,
                        help="Balanced dummy TRs used to reach the steady state.")
    parser.add_argument("--fov-mm", type=float, default=20.0, help="Field of view (mm).")
    parser.add_argument("--output", type=Path, help="Optional output PNG path.")
    return parser.parse_args()


def _phantom(n: int) -> np.ndarray:
    """A disk with a couple of bars, so banding and shading are easy to read."""

    rho = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.mgrid[0:n, 0:n]
    rho[(xx - n * 0.5) ** 2 + (yy - n * 0.5) ** 2 <= (n * 0.38) ** 2] = 1.0
    for col in range(int(n * 0.30), int(n * 0.70), 5):
        rho[(xx >= col) & (xx < col + 2) & (yy >= n * 0.30) & (yy < n * 0.70)] = 0.0
    return rho


def _smooth_b0_map(n: int, peak_hz: float) -> np.ndarray:
    """A smooth low-order B0 residual (linear + saddle), in rad/s."""

    u = np.linspace(-1.0, 1.0, n)
    xx, zz = np.meshgrid(u, u, indexing="ij")
    shape = 0.6 * zz + 0.5 * (xx**2 - zz**2)  # tilt + saddle, like a shim residual
    shape = shape / np.max(np.abs(shape))
    return 2.0 * np.pi * float(peak_hz) * shape


def _panel_image(ax, image, rho, title):
    img = np.abs(image)
    ref = np.abs(rho).max() or 1.0
    scale = img[rho > 0].mean() if np.any(rho > 0) else img.max()
    vmax = (scale / max(ref, 1e-9)) * 1.6 if scale > 0 else 1.0
    ax.imshow(img.T, origin="lower", cmap="gray", vmin=0.0, vmax=max(vmax, 1e-9))
    ax.set_title(title, fontsize="medium")
    ax.set_xticks([])
    ax.set_yticks([])


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.workflows import (
        bssfp_optimal_flip_deg,
        bssfp_steady_state_signal,
        run_bssfp_imaging,
    )

    n = int(args.pixels)
    t1 = args.t1_ms * 1e-3
    t2 = args.t2_ms * 1e-3
    readout = args.readout_time_ms * 1e-3
    fov = (args.fov_mm * 1e-3, args.fov_mm * 1e-3)
    phase_time = 0.5e-3
    excitation = 0.5e-3
    tr = excitation + 2.0 * phase_time + readout
    band_hz = 1.0 / tr
    common = dict(
        fov=fov, flip_angle_deg=args.flip_deg, readout_time=readout,
        phase_time=phase_time, excitation_duration=excitation,
        num_dummy_tr=int(args.num_dummy_tr),
    )

    rho = _phantom(n)
    t1_map = np.full((n, n), t1)
    t2_map = np.full((n, n), t2)
    b0_map = _smooth_b0_map(n, args.b0_inhomogeneity_hz)
    b1_map = np.linspace(args.b1_min, args.b1_max, n)[:, np.newaxis] * np.ones((1, n))

    print(f"bSSFP {n}x{n}: TR={tr*1e3:.2f} ms -> band spacing 1/TR = {band_hz:.0f} Hz")
    print(f"T1={args.t1_ms:.0f} ms, T2={args.t2_ms:.0f} ms, flip={args.flip_deg:.0f} deg "
          f"(optimal {bssfp_optimal_flip_deg(t1, t2):.0f} deg)")
    print(f"B0 inhomogeneity +/-{args.b0_inhomogeneity_hz:.0f} Hz "
          f"(~{args.b0_inhomogeneity_hz/band_hz:.1f} bands across the FOV); "
          f"B1 {args.b1_min:.2f}-{args.b1_max:.2f}")

    # Off-resonance banding profile (gradient-free, same engine primitives).
    df = np.linspace(-1.6 * band_hz, 1.6 * band_hz, 401)
    prof_180 = np.abs(bssfp_steady_state_signal(
        df, flip_angle_deg=args.flip_deg, tr=tr, t1=t1, t2=t2, phase_increment_deg=180.0))
    prof_0 = np.abs(bssfp_steady_state_signal(
        df, flip_angle_deg=args.flip_deg, tr=tr, t1=t1, t2=t2, phase_increment_deg=0.0))

    # Signal vs flip angle at band centre, for two T2/T1 contrasts.
    flips = np.linspace(5.0, 120.0, 40)
    sig_a = np.abs([bssfp_steady_state_signal(
        0.0, flip_angle_deg=f, tr=tr, t1=t1, t2=t2, phase_increment_deg=180.0) for f in flips]).ravel()
    t2b = 0.5 * t2
    sig_b = np.abs([bssfp_steady_state_signal(
        0.0, flip_angle_deg=f, tr=tr, t1=t1, t2=t2b, phase_increment_deg=180.0) for f in flips]).ravel()

    print("imaging: reference, B0 banding, B1 shading, phase-cycled pair...")
    reference = run_bssfp_imaging(rho, t1_map=t1_map, t2_map=t2_map, **common)
    banded = run_bssfp_imaging(rho, t1_map=t1_map, t2_map=t2_map, b0_map=b0_map, **common)
    shaded = run_bssfp_imaging(rho, t1_map=t1_map, t2_map=t2_map, b1_tx_map=b1_map, **common)
    banded_pc = run_bssfp_imaging(
        rho, t1_map=t1_map, t2_map=t2_map, b0_map=b0_map, phase_increment_deg=0.0,
        fov=fov, flip_angle_deg=args.flip_deg, readout_time=readout,
        phase_time=phase_time, excitation_duration=excitation,
        num_dummy_tr=int(args.num_dummy_tr))
    combined = np.maximum(np.abs(banded.magnitude[:, :, 0]), np.abs(banded_pc.magnitude[:, :, 0]))

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.8), constrained_layout=True)

    ax[0, 0].plot(df, prof_180, color="C0", label=r"$\Delta\phi=180^\circ$")
    ax[0, 0].plot(df, prof_0, color="C1", label=r"$\Delta\phi=0^\circ$")
    for k in (-1, 0, 1):
        ax[0, 0].axvline((k + 0.5) * band_hz, color="C0", ls=":", lw=0.7)
        ax[0, 0].axvline(k * band_hz, color="C1", ls=":", lw=0.7)
    ax[0, 0].set_xlabel("off-resonance (Hz)")
    ax[0, 0].set_ylabel(r"steady-state $|M_{xy}|$")
    ax[0, 0].set_title(f"Off-resonance response (band 1/TR = {band_hz:.0f} Hz)")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(flips, sig_a, color="C2", label=f"T2/T1={t2/t1:.2f}")
    ax[0, 1].plot(flips, sig_b, color="C3", label=f"T2/T1={t2b/t1:.2f}")
    for tt, col in [(t2, "C2"), (t2b, "C3")]:
        astar = bssfp_optimal_flip_deg(t1, tt)
        ax[0, 1].axvline(astar, color=col, ls=":", lw=0.8)
        ax[0, 1].axhline(0.5 * np.sqrt(tt / t1), color=col, ls="--", lw=0.6)
    ax[0, 1].set_xlabel("flip angle (deg)")
    ax[0, 1].set_ylabel(r"band-centre $|M_{xy}|$")
    ax[0, 1].set_title(r"Signal vs flip (peak $\approx M_0/2\,\sqrt{T_2/T_1}$)")
    ax[0, 1].legend(fontsize=8)

    _panel_image(ax[0, 2], reference.magnitude[:, :, 0], rho, "reference (uniform fields)")
    _panel_image(ax[1, 0], banded.magnitude[:, :, 0], rho,
                 f"B0 +/-{args.b0_inhomogeneity_hz:.0f} Hz -> banding")
    _panel_image(ax[1, 1], shaded.magnitude[:, :, 0], rho,
                 f"B1 {args.b1_min:.2f}-{args.b1_max:.2f} -> shading")
    _panel_image(ax[1, 2], combined, rho, "phase-cycled 180+0 (bands removed, 2x time)")

    fig.suptitle("Balanced SSFP: high steady-state SNR, but sensitive to B0/B1 inhomogeneity")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
