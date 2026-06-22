"""Spin-3/2 powder NQR nutation curve from the full density-matrix model.

A nutation experiment varies the excitation pulse length and records the signal
amplitude. For a powder the RF coupling depends on each crystallite's
orientation, so the curve peaks at a flip angle larger than 90 degrees. The
classic spin-1 result peaks near 119 degrees; the four-state spin-3/2 model
peaks at a slightly smaller flip angle.

This example uses ``spin_dynamics.nqr.full_dynamics`` -- the full ``(2I+1)``
density matrix required for spin-3/2 -- and overlays the spin-1 curve (computed
the same way) as a validation anchor against the 119-degree reference.

The flip-angle axis is calibrated experimentally: the best-coupled crystallite's
first signal maximum defines 90 degrees, exactly as a measured nutation curve is
referenced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    detection_operator,
    diagonalize_site,
    equilibrium_density,
    powder_average_grid,
    pulse_hamiltonian,
)


def powder_nutation(site, *, nutation_hz, times, orientations, carrier_hz=None):
    """Return the complex powder signal and per-orientation signals over `times`.

    Levels are orientation-independent in the EFG principal-axis frame, so the
    site is diagonalized once; only the RF/detection direction changes per
    crystallite. For each orientation the pulse propagator is diagonalized once
    and the signal over all pulse lengths is evaluated as a sum of complex
    exponentials ``sum_{j,k} A_jk B_kj exp(-i (lam_j - lam_k) t)``.
    """

    eigensystem = diagonalize_site(site)
    if carrier_hz is None:
        carrier_hz = max(t.frequency_hz for t in eigensystem.transitions)
    rho0 = equilibrium_density(eigensystem.levels_hz)
    times = np.asarray(times, dtype=np.float64)

    per_orientation = np.zeros((len(orientations), times.size), dtype=np.complex128)
    weights = np.array([o.weight for o in orientations], dtype=np.float64)
    for index, orientation in enumerate(orientations):
        b1 = orientation.b1_direction_pas
        hamiltonian = pulse_hamiltonian(
            eigensystem, nutation_hz=nutation_hz, rf_frequency_hz=carrier_hz,
            phase=0.0, b1_direction_pas=b1,
        )
        detector = detection_operator(eigensystem, carrier_hz, b1)
        lam, vec = np.linalg.eigh(hamiltonian)
        a_mat = vec.conj().T @ rho0 @ vec
        b_mat = vec.conj().T @ detector @ vec
        coeff = a_mat * b_mat.T  # coeff[j, k] = A[j, k] * B[k, j]
        freq = lam[:, None] - lam[None, :]
        per_orientation[index] = (
            coeff[:, :, None] * np.exp(-1j * freq[:, :, None] * times[None, None, :])
        ).sum(axis=(0, 1))
    powder = weights @ per_orientation
    return powder, per_orientation


def _first_peak_time(magnitude: np.ndarray, times: np.ndarray) -> float:
    """Return the time of the first interior local maximum of `magnitude`."""

    interior = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
    indices = np.nonzero(interior)[0]
    if indices.size:
        return float(times[indices[0] + 1])
    return float(times[int(np.argmax(magnitude))])


def nutation_curve(site, *, nutation_hz, orientations, max_angle_deg, points):
    """Return (flip_angle_deg, normalized_amplitude, peak_angle_deg)."""

    # First pass on a generous time grid to locate the best-coupled crystallite's
    # 90-degree time, then relabel the axis in degrees.
    coupling_scale = 2.0 * np.pi * nutation_hz * float(site.spin)  # rough Rabi scale
    t_probe = np.linspace(0.0, 4.0 * (np.pi / 2) / max(coupling_scale, 1.0), 400)
    _, per = powder_nutation(site, nutation_hz=nutation_hz, times=t_probe,
                             orientations=orientations)
    best = int(np.argmax(np.max(np.abs(per), axis=1)))
    t_90 = _first_peak_time(np.abs(per[best]), t_probe)

    times = np.linspace(0.0, (max_angle_deg / 90.0) * t_90, points)
    powder, _ = powder_nutation(site, nutation_hz=nutation_hz, times=times,
                                orientations=orientations)
    angles = 90.0 * times / t_90
    magnitude = np.abs(powder)
    peak_angle = float(angles[int(np.argmax(magnitude))])
    norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
    return angles, norm, peak_angle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrupole-mhz", type=float, default=1.0,
                        help="Quadrupole line frequency parameter in MHz.")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="EFG asymmetry parameter.")
    parser.add_argument("--nutation-khz", type=float, default=1.0,
                        help="Bare RF nutation gamma*B1/(2*pi) in kHz. Keep it well "
                             "below the line frequency: the spin-1 peak converges to "
                             "the 119-degree reference as the pulse weakens.")
    parser.add_argument("--max-angle", type=float, default=270.0,
                        help="Maximum flip angle in degrees.")
    parser.add_argument("--points", type=int, default=181, help="Flip-angle samples.")
    parser.add_argument("--n-theta", type=int, default=16, help="Powder polar samples.")
    parser.add_argument("--n-phi", type=int, default=32, help="Powder azimuthal samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    orientations = powder_average_grid(args.n_theta, args.n_phi)
    nutation_hz = args.nutation_khz * 1e3

    spin32 = QuadrupolarSite(spin=1.5, isotope="35Cl",
                             quadrupole_frequency_hz=args.quadrupole_mhz * 1e6,
                             eta=args.eta)
    spin1 = QuadrupolarSite(spin=1, isotope="14N",
                            quadrupole_frequency_hz=args.quadrupole_mhz * 1e6,
                            eta=max(args.eta, 0.2))

    a32, s32, peak32 = nutation_curve(spin32, nutation_hz=nutation_hz,
                                      orientations=orientations,
                                      max_angle_deg=args.max_angle, points=args.points)
    a1, s1, peak1 = nutation_curve(spin1, nutation_hz=nutation_hz,
                                   orientations=orientations,
                                   max_angle_deg=args.max_angle, points=args.points)

    print(f"spin-1   powder nutation peak: {peak1:6.1f} deg (reference ~119 deg)")
    print(f"spin-3/2 powder nutation peak: {peak32:6.1f} deg")
    print(f"spin-3/2 peak is {peak1 - peak32:.1f} deg smaller than spin-1")

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(a1, s1, color="C0", label=f"spin-1 (peak {peak1:.0f} deg)")
    ax.plot(a32, s32, color="C3", label=f"spin-3/2 (peak {peak32:.0f} deg)")
    ax.axvline(119.0, color="0.5", linewidth=1, linestyle="--", alpha=0.7,
               label="119 deg reference")
    ax.axvline(peak1, color="C0", linewidth=1, alpha=0.4)
    ax.axvline(peak32, color="C3", linewidth=1, alpha=0.4)
    ax.set_xlabel("Flip angle of best-coupled crystallite (degrees)")
    ax.set_ylabel("Normalized powder signal amplitude")
    ax.set_title("Powder NQR nutation (full density-matrix model)")
    ax.legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
