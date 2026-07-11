"""Circular vs linear RF polarization for powder 14N NQR (SLSE detection).

Every other NQR example in this package excites the powder with a single linear
RF coil. Here we relax that: a *circularly* polarized drive uses two orthogonal
coils driven pi/2 out of phase (a rotating RF field), with matched quadrature
detection. In zero-field NQR the transition has no intrinsic precession
handedness, so a linear coil couples equally to both rotation senses while a
circular coil couples to one -- and across a powder the two schemes average
differently.

The classic result is that circular polarization slightly raises the *refocused*
(spin-locked spin-echo, SLSE) powder signal. This example confirms and quantifies
it with the full ``(2I+1)`` density-matrix model:

* Neither circular excitation alone (single-coil detection) nor quadrature
  detection alone helps; **together** they raise the refocused powder signal by
  ~1.6x at matched per-coil B1.
* Detecting with the *wrong* rotation sense nulls the signal -- a direct check
  that the handedness is physical, not a normalization artifact.
* Circular uses two coils, so it draws ~2x peak RF power and adds a sqrt(2)
  two-channel receiver-noise penalty; the ~1.6x signal gain becomes a ~1.1x net
  SNR gain -- the "slight increase" reported in the literature.

Because multi-coil coupling depends on the *full* crystallite orientation (not
just one coil axis), the powder average runs over an SO(3) frame grid
(``powder_frame_grid``). The flip-angle axis is referenced to the best-coupled
crystallite's 90 degrees (calibrated from the linear scheme, as a measured
nutation curve is), so the classic spin-1 powder maximum sits near 119 degrees
and circular's greater RF efficiency shows as an earlier peak. The ~1.6x signal
gain compares each scheme at its own optimal flip at matched per-coil B1, which
isolates the polarization effect from the trivial ~2x field efficiency of a
rotating field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    CoilDrive,
    QuadrupolarSite,
    detection_operator,
    diagonalize_site,
    equilibrium_density,
    multi_axis_pulse_hamiltonian,
    powder_frame_grid,
    quadrature_detection_operator,
    static_hamiltonian_rotating,
)


def _eig(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(hamiltonian)


def _first_peak_time(magnitude: np.ndarray, times: np.ndarray) -> float:
    """Return the time of the first interior local maximum of ``magnitude``."""

    interior = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
    indices = np.nonzero(interior)[0]
    if indices.size:
        return float(times[indices[0] + 1])
    return float(times[int(np.argmax(magnitude))])


def _u(lam: np.ndarray, vec: np.ndarray, t: float) -> np.ndarray:
    """Reconstruct ``exp(-i H t)`` from a cached Hermitian eigendecomposition."""

    return (vec * np.exp(-1j * lam * t)[np.newaxis, :]) @ vec.conj().T


class _SchemePrep:
    """Cached per-frame propagator ingredients for one polarization scheme."""

    def __init__(self, eig, carrier, nutation_hz, frames, *, scheme, helicity=1,
                 detect_helicity=None):
        self.free_omega = np.real(np.diag(static_hamiltonian_rotating(eig, carrier)))
        self.rho0 = equilibrium_density(eig.levels_hz)
        self.weights = np.array([f.weight for f in frames], dtype=np.float64)
        self.ex = []   # (lam, vec) for the phase-0 excitation pulse
        self.ref = []  # (lam, vec) for the phase-pi/2 refocusing pulse
        self.det = []  # detection observable
        detect_helicity = helicity if detect_helicity is None else detect_helicity
        for frame in frames:
            a1, a2 = frame.x, frame.y
            if scheme == "linear":
                ex_coils = [CoilDrive(nutation_hz, 0.0, tuple(a1))]
                ref_coils = [CoilDrive(nutation_hz, np.pi / 2.0, tuple(a1))]
                det = detection_operator(eig, carrier, a1)
            else:  # circular: two orthogonal coils in quadrature
                ex_coils = [
                    CoilDrive(nutation_hz, 0.0, tuple(a1)),
                    CoilDrive(nutation_hz, helicity * np.pi / 2.0, tuple(a2)),
                ]
                ref_coils = [
                    CoilDrive(nutation_hz, np.pi / 2.0, tuple(a1)),
                    CoilDrive(nutation_hz, np.pi / 2.0 + helicity * np.pi / 2.0, tuple(a2)),
                ]
                det = quadrature_detection_operator(
                    eig, carrier, a1, a2, helicity=detect_helicity
                )
            hx = multi_axis_pulse_hamiltonian(eig, ex_coils, rf_frequency_hz=carrier)
            hr = multi_axis_pulse_hamiltonian(eig, ref_coils, rf_frequency_hz=carrier)
            self.ex.append(_eig(hx))
            self.ref.append(_eig(hr))
            self.det.append(det)
        self.n = len(frames)

    def best_coupled_t90(self, nutation_hz: float) -> float:
        """Calibrate 90 degrees from the best-coupled crystallite's first FID max.

        The flip-angle axis of a measured nutation curve is referenced to the
        best-coupled crystallite (the one whose transition dipole aligns with the
        coil), exactly as ``plot_nqr_full_powder_nutation`` does. The realized
        Rabi rate is ``2 * nutation * |<a|e.I|b>|``, so this experimental
        calibration -- rather than the bare ``gamma * B1`` nominal -- puts the
        classic spin-1 powder maximum near 119 degrees.
        """

        t_probe = np.linspace(0.0, 1.0 / (2.0 * nutation_hz), 240)
        best_time = t_probe[-1]
        best_amp = -1.0
        for k in range(self.n):
            lam, vec = self.ex[k]
            det = self.det[k]
            mag = np.empty(t_probe.size)
            for i, t in enumerate(t_probe):
                u = _u(lam, vec, t)
                rho = u @ self.rho0 @ u.conj().T
                mag[i] = abs(np.trace(rho @ det))
            if mag.max() > best_amp:
                best_amp = mag.max()
                best_time = _first_peak_time(mag, t_probe)
        return best_time

    def fid(self, durations: np.ndarray) -> np.ndarray:
        """Powder complex FID amplitude vs excitation duration."""

        out = np.zeros(durations.size, dtype=np.complex128)
        for k in range(self.n):
            lam, vec = self.ex[k]
            det = self.det[k]
            w = self.weights[k]
            for i, dur in enumerate(durations):
                u = _u(lam, vec, dur)
                rho = u @ self.rho0 @ u.conj().T
                out[i] += w * np.trace(rho @ det)
        return out

    def slse(self, durations: np.ndarray, ref_factor: float, echo_spacing: float,
             num_echoes: int) -> np.ndarray:
        """Powder SLSE echo-train (num_echoes x len(durations)) complex amplitude."""

        out = np.zeros((num_echoes, durations.size), dtype=np.complex128)
        for k in range(self.n):
            lam_e, vec_e = self.ex[k]
            lam_r, vec_r = self.ref[k]
            det = self.det[k]
            w = self.weights[k]
            for i, dur in enumerate(durations):
                ref_dur = ref_factor * dur
                free_half = 0.5 * (echo_spacing - ref_dur)
                if free_half < 0:
                    continue
                free_phase = np.exp(-1j * self.free_omega * free_half)
                u_ex = _u(lam_e, vec_e, dur)
                u_ref = _u(lam_r, vec_r, ref_dur)
                rho = u_ex @ self.rho0 @ u_ex.conj().T
                for e in range(num_echoes):
                    rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
                    rho = u_ref @ rho @ u_ref.conj().T
                    rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
                    out[e, i] += w * np.trace(rho @ det)
        return out

    def slse_per_frame(self, dur: float, ref_factor: float, echo_spacing: float,
                       num_echoes: int) -> np.ndarray:
        """First-echo magnitude per frame at a single excitation duration."""

        vals = np.zeros(self.n, dtype=np.complex128)
        ref_dur = ref_factor * dur
        free_half = 0.5 * (echo_spacing - ref_dur)
        free_phase = np.exp(-1j * self.free_omega * free_half)
        for k in range(self.n):
            lam_e, vec_e = self.ex[k]
            lam_r, vec_r = self.ref[k]
            det = self.det[k]
            u_ex = _u(lam_e, vec_e, dur)
            u_ref = _u(lam_r, vec_r, ref_dur)
            rho = u_ex @ self.rho0 @ u_ex.conj().T
            rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
            rho = u_ref @ rho @ u_ref.conj().T
            rho = (free_phase[:, None] * rho) * free_phase.conj()[None, :]
            vals[k] = np.trace(rho @ det)
        return np.abs(vals)


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quadrupole-mhz", type=float, default=3.0,
                        help="Quadrupole line-frequency parameter in MHz.")
    parser.add_argument("--eta", type=float, default=0.3, help="EFG asymmetry parameter.")
    parser.add_argument("--nutation-khz", type=float, default=20.0,
                        help="Per-coil bare nutation gamma*B1/(2*pi) in kHz.")
    parser.add_argument("--echo-spacing-us", type=float, default=200.0,
                        help="SLSE echo spacing in microseconds.")
    parser.add_argument("--num-echoes", type=int, default=6, help="SLSE echoes per train.")
    parser.add_argument("--ref-factor", type=float, default=2.0,
                        help="Refocus duration as a multiple of the excitation duration.")
    parser.add_argument("--max-flip", type=float, default=360.0,
                        help="Maximum realized flip angle (degrees, best-coupled "
                             "crystallite) on the sweep.")
    parser.add_argument("--points", type=int, default=49, help="Flip-angle samples.")
    parser.add_argument("--n-theta", type=int, default=8, help="Powder polar nodes.")
    parser.add_argument("--n-phi", type=int, default=14, help="Powder azimuthal nodes.")
    parser.add_argument("--n-chi", type=int, default=6, help="Powder in-plane nodes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(spin=1, isotope="14N",
                           quadrupole_frequency_hz=args.quadrupole_mhz * 1e6,
                           eta=args.eta)
    eig = diagonalize_site(site)
    carrier = max(t.frequency_hz for t in eig.transitions)  # highest line: valid RWA
    nutation = args.nutation_khz * 1e3
    frames = powder_frame_grid(args.n_theta, args.n_phi, args.n_chi)
    echo_spacing = args.echo_spacing_us * 1e-6

    linear = _SchemePrep(eig, carrier, nutation, frames, scheme="linear")
    circ = _SchemePrep(eig, carrier, nutation, frames, scheme="circular", helicity=1)
    circ_wrong = _SchemePrep(eig, carrier, nutation, frames, scheme="circular",
                             helicity=1, detect_helicity=-1)

    # Reference the flip-angle axis to the best-coupled crystallite's 90 degrees,
    # calibrated experimentally from the linear scheme (the standard nutation-curve
    # convention, as in plot_nqr_full_powder_nutation). Both schemes share this
    # time-to-angle map, so the classic spin-1 powder maximum lands near 119 deg
    # and circular's greater RF efficiency shows as an earlier peak. Using the bare
    # gamma*B1 nominal instead would be a factor ~2 too small for spin-1 NQR.
    t90 = linear.best_coupled_t90(nutation)
    flip_deg = np.linspace(0.0, args.max_flip, args.points)
    durations = (flip_deg / 90.0) * t90

    # FID nutation curves
    fid_lin = np.abs(linear.fid(durations))
    fid_cir = np.abs(circ.fid(durations))

    # SLSE refocused-signal curves (first-echo magnitude vs flip)
    slse_lin = linear.slse(durations, args.ref_factor, echo_spacing, args.num_echoes)
    slse_cir = circ.slse(durations, args.ref_factor, echo_spacing, args.num_echoes)
    slse_cir_wrong = circ_wrong.slse(durations, args.ref_factor, echo_spacing,
                                     args.num_echoes)
    echo1_lin = np.abs(slse_lin[0])
    echo1_cir = np.abs(slse_cir[0])
    echo1_cir_wrong = np.abs(slse_cir_wrong[0])
    energy_lin = np.abs(slse_lin).sum(axis=0)   # train energy vs flip
    energy_cir = np.abs(slse_cir).sum(axis=0)

    # Best-vs-best (each scheme at its own optimal flip)
    i_lin = int(np.argmax(energy_lin))
    i_cir = int(np.argmax(energy_cir))
    best_lin, best_cir = energy_lin[i_lin], energy_cir[i_cir]
    signal_ratio = best_cir / best_lin if best_lin > 0 else np.nan
    snr_ratio = signal_ratio / np.sqrt(2.0)  # sqrt(2) two-channel receiver noise

    # Per-orientation uniformity at each scheme's optimum
    unif_lin = linear.slse_per_frame(durations[i_lin], args.ref_factor, echo_spacing,
                                     args.num_echoes)
    unif_cir = circ.slse_per_frame(durations[i_cir], args.ref_factor, echo_spacing,
                                   args.num_echoes)

    print("Circular vs linear RF polarization -- powder 14N NQR (SLSE detection)")
    print(f"nu_Q parameter: {args.quadrupole_mhz:.3f} MHz   eta: {args.eta:.2f}   "
          f"carrier: {carrier / 1e6:.4f} MHz")
    print(f"per-coil nutation: {args.nutation_khz:.0f} kHz   echoes: {args.num_echoes}   "
          f"orientations: {len(frames)} (SO(3) grid)")
    print()
    print("(flip axis is the realized best-coupled-crystallite angle: "
          "2 x nutation x matrix element; classic spin-1 powder max ~119 deg)")
    dur_lin_us = durations[i_lin] * 1e6
    dur_cir_us = durations[i_cir] * 1e6
    print(f"linear   optimum: flip {flip_deg[i_lin]:5.1f} deg   "
          f"pulse {dur_lin_us:5.2f} us   train energy {best_lin:.3f}")
    print(f"circular optimum: flip {flip_deg[i_cir]:5.1f} deg   "
          f"pulse {dur_cir_us:5.2f} us   train energy {best_cir:.3f}")
    print(f"circular reaches its optimum at {dur_lin_us / dur_cir_us:.2f}x shorter "
          f"pulse (rotating field is more RF-efficient)")
    print()
    print(f"refocused-signal gain (circular / linear, each at its own optimum): "
          f"{signal_ratio:.2f}x")
    print(f"net SNR gain after sqrt(2) two-channel noise: {snr_ratio:.2f}x "
          f"(circular draws ~2x peak RF power)")
    print(f"wrong-sense quadrature detection first echo: "
          f"{echo1_cir_wrong.max():.4f}  (matched: {echo1_cir.max():.4f}) -- handedness check")
    print(f"per-orientation SLSE spread (min/max): linear {unif_lin.min()/unif_lin.max():.3f}   "
          f"circular {unif_cir.min()/unif_cir.max():.3f}")

    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(flip_deg, fid_lin / fid_lin.max(), color="C0", label="linear (1 coil)")
    ax.plot(flip_deg, fid_cir / fid_lin.max(), color="C3", label="circular (2 coils, quad)")
    ax.set_xlabel("Realized flip angle, best-coupled crystallite (deg)")
    ax.set_ylabel("Powder FID amplitude (norm. to linear peak)")
    ax.set_title("(a) Single-pulse FID nutation")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(flip_deg, echo1_lin, color="C0", label="linear")
    ax.plot(flip_deg, echo1_cir, color="C3", label="circular + matched quad RX")
    ax.axvline(flip_deg[i_lin], color="C0", ls=":", alpha=0.5)
    ax.axvline(flip_deg[i_cir], color="C3", ls=":", alpha=0.5)
    ax.set_xlabel("Realized flip angle, best-coupled crystallite (deg)")
    ax.set_ylabel("Refocused SLSE first-echo amplitude")
    ax.set_title(f"(b) SLSE refocused signal ({signal_ratio:.2f}x at optima)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(flip_deg, echo1_cir, color="C3", label="matched sense (+)")
    ax.plot(flip_deg, echo1_cir_wrong, color="0.5", label="wrong sense (-)")
    ax.set_xlabel("Realized flip angle, best-coupled crystallite (deg)")
    ax.set_ylabel("Circular SLSE first-echo amplitude")
    ax.set_title("(c) Detection handedness: wrong sense nulls")
    ax.legend()

    ax = axes[1, 1]
    bins = np.linspace(0, max(unif_lin.max(), unif_cir.max()), 24)
    ax.hist(unif_lin, bins=bins, color="C0", alpha=0.6, label="linear", weights=linear.weights)
    ax.hist(unif_cir, bins=bins, color="C3", alpha=0.6, label="circular", weights=circ.weights)
    ax.set_xlabel("Per-orientation SLSE first-echo magnitude")
    ax.set_ylabel("Powder weight")
    ax.set_title("(d) Excitation uniformity across the powder")
    ax.legend()

    fig.suptitle("Circular vs linear RF polarization -- powder 14N NQR", fontsize=13)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"\nsaved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
