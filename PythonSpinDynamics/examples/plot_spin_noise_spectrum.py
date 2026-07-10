"""Physical spin-noise spectra: circuit model and stochastic simulation.

Demonstrates the two spin-noise models documented in ``docs/spin_noise.md``:

* **Panel 1 (Option B, frequency domain).** Tuned-probe output noise density
  with the Hoult sample impedance ``Z_n`` in series with the coil, for a
  sample at the coil temperature (equilibrium dip), a cold sample (deeper
  dip), and hot spins (bump). The coil Johnson noise stays at the circuit
  temperature while the sample emits at its own temperature.
* **Panel 2 (Option C, time domain).** Monte-Carlo spectrum of the
  receiver-node signal from the stochastic Langevin model
  (``simulate_spin_noise``) at thermal equilibrium, against the analytic
  linear-response curve ``spin_noise_output_psd`` -- including the
  radiation-damping-broadened equilibrium dip ``S(0)/S_e = Trd/(T2+Trd)``.
* **Panel 3.** Spin-noise-seeded NMR maser ignition: an inverted, pumped
  sample above threshold starting from *zero* transverse magnetization; the
  stochastic fluctuations alone ignite the maser.

Both models share one coupling constant with the deterministic
radiation-damping module via ``R_n0 = R_coil * T2 / (2 * Trd)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.noise import tuned_probe_output_noise_density  # noqa: E402
from spin_dynamics.radiation_damping import (  # noqa: E402
    KB,
    MU0,
    PROTON_GAMMA,
    RadiationDampingProbe,
)
from spin_dynamics.spin_noise import (  # noqa: E402
    SampleCoupling,
    SpinNoiseSource,
    estimate_spin_noise_spectrum,
    sample_coupling_from_radiation_damping,
    simulate_spin_noise,
    spin_noise_output_psd,
)


def probe_with_trd(trd_target: float, *, q: float = 50.0, fill: float = 0.5) -> RadiationDampingProbe:
    m0 = 2.0 / (PROTON_GAMMA * MU0 * fill * q * trd_target)
    return RadiationDampingProbe(
        gamma=PROTON_GAMMA,
        omega0=2 * np.pi * 1e6,
        q=q,
        fill_factor=fill,
        equilibrium_magnetization=m0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=60.0, help="Sim length (s).")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    # ------------------------------------------------------------------
    # Panel 1: Option B -- tuned-probe density with the sample impedance.
    # ------------------------------------------------------------------
    f0 = 1e6
    w0 = 2 * np.pi * f0
    L = 10e-6
    q = 50.0
    t_90 = 25e-6
    w1_max = (np.pi / 2) / t_90
    sp = {
        "k": KB,
        "T": 300.0,
        "L": L,
        "R": w0 * L / q,
        "C": 1 / (w0**2 * L),
        "Cin": 1e-15,
        "Rin": 1e12,
        "Rd": 1e12,
        "vn": 0.0,
        "in_": 0.0,
        "w0": w0,
        "del_w": np.linspace(-0.2, 0.2, 8001),
    }
    pp = {"T_90": t_90}
    t2_line = 400.0 / w1_max
    r_coil = w0 * L / q
    trd = 2.0 * 0.02  # illustrative Trd for the circuit panel
    r_n0 = sample_coupling_from_radiation_damping(
        trd=trd, coil_resistance=r_coil, t2=t2_line, temperature=300.0
    ).r_n0
    base, freqs_b = tuned_probe_output_noise_density(sp, pp)
    cases = {
        "sample at coil T (dip)": 300.0,
        "cold sample 4 K": 4.0,
        "hot spins 100x": 30000.0,
    }
    print("Option B: tuned circuit with sample impedance")
    print(f"  R_coil = {r_coil:.3f} ohm, R_n0 = {r_n0:.3f} ohm, T2 = {t2_line*1e3:.2f} ms")
    densities = {}
    for label, temp in cases.items():
        coupling = SampleCoupling(r_n0=r_n0, t2=t2_line, temperature=temp)
        dens, _ = tuned_probe_output_noise_density(sp, pp, sample=coupling)
        densities[label] = dens
        center = len(sp["del_w"]) // 2
        print(f"  {label}: on-resonance / baseline-free ratio = {dens[center]/base[center]:.3f}")

    # ------------------------------------------------------------------
    # Panels 2-3: Option C -- stochastic simulation.
    # ------------------------------------------------------------------
    probe = probe_with_trd(0.02)
    t2 = 0.01
    source = SpinNoiseSource(
        m_rms=1e-5, t2=t2, sample_temperature=300.0, coil_temperature=300.0
    )
    dt = 5e-4
    time = np.arange(0.0, args.duration, dt)
    res = simulate_spin_noise(time, probe, source, seed=args.seed)
    gamma_rate = 1.0 / t2 + 1.0 / probe.trd
    burn = int(5 * max(t2, probe.trd) / dt)
    block = int(round(12.0 / gamma_rate / dt))
    freqs_c, psd = estimate_spin_noise_spectrum(res.emf[burn:], dt, block_samples=block)
    analytic = spin_noise_output_psd(freqs_c, source=source, trd=probe.trd)
    print("Option C: stochastic model at equilibrium")
    print(f"  Trd = {probe.trd*1e3:.1f} ms, T2 = {t2*1e3:.1f} ms")
    print(f"  analytic dip S(0)/S_e = {probe.trd/(t2+probe.trd):.3f}")

    maser_probe = probe_with_trd(0.002)
    maser_source = SpinNoiseSource(
        m_rms=1e-6, t2=t2, sample_temperature=300.0, coil_temperature=0.0
    )
    maser_time = np.arange(0.0, 0.4, 1e-4)
    maser = simulate_spin_noise(
        maser_time,
        maser_probe,
        maser_source,
        initial_mxy=0.0 + 0.0j,
        initial_mz=-1.0,
        t1=0.05,
        equilibrium_mz=-1.0,
        seed=args.seed,
    )
    print(f"  maser peak |mxy| from pure spin noise: {np.max(np.abs(maser.mxy)):.3g}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    offs_hz = freqs_b - f0
    for label, dens in densities.items():
        ax.semilogy(offs_hz, dens / base, label=label)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="no sample")
    ax.set_xlim(-300, 300)
    ax.set_xlabel("offset from carrier (Hz)")
    ax.set_ylabel("density / no-sample baseline")
    ax.set_title("Option B: sample impedance in tuned circuit")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(freqs_c, psd, lw=0.8, label="simulated (emf)")
    ax.plot(freqs_c, analytic, "k--", label="analytic linear response")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("offset (Hz)")
    ax.set_ylabel("PSD (norm. units$^2$/Hz)")
    ax.set_title("Option C: equilibrium dip, $T_c = T_s$")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogy(maser_time, np.abs(maser.mxy) + 1e-12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|mxy| (normalized)")
    ax.set_title("Spin-noise-seeded maser ignition")
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
