r"""Quantify SLSE vs. steady-state SORC NQR sensitivity per unit time.

Two multiple-pulse NQR detection schemes trade off very differently against the
relaxation times:

* **SLSE** (spin-lock spin echo) excites the full equilibrium magnetization,
  reads a decaying train of ``K`` echoes (envelope time ``T2e``), and then must
  wait for longitudinal recovery (``~T1``) before the next scan. When ``T1`` is
  long, most of the experiment is spent waiting.
* **SORC** (strong off-resonance comb) pulses continuously and settles into a
  *steady state*: every cycle emits signal and there is no recovery wait, so the
  duty cycle stays near unity regardless of ``T1``.

This example makes the standard sensitivity argument quantitative and checks it
against the package's density-matrix / dephased-pathway NQR simulators, for both
a single crystal and a powder.

Mathematical analysis
---------------------
The figure of merit is SNR per unit square-root of measurement time, because
averaging ``N`` acquisitions improves SNR as ``sqrt(N)`` and ``N = T / t_scan``.
Take white receiver noise with the same per-observation RMS ``sigma`` for both
schemes (matched acquisition bandwidth), and let a single equilibrium pulse give
signal amplitude ``a0``.

*SLSE.* One scan yields echoes ``a_k = A M e^{-k t_E / T2e}``, so its matched
filter collects energy ``E = sqrt(sum_k |a_k|^2) ~ A M sqrt(T2e / 2 t_E)`` for
``t_E << T2e``. With repetition time ``T_R`` the recovered magnetization is
``M = M0 (1 - e^{-T_R/T1})`` and ``N = T/T_R`` scans average, giving

    S_SLSE(T_R) ~ E(K) * (1 - e^{-T_R/T1}) / sqrt(T_R),   T_R >= K t_E.

Maximizing ``(1 - e^{-x})/sqrt(x)`` (peak at ``x* ~= 1.2564``, value ``0.6383``)
gives the classic ``1/sqrt(T1)`` penalty, ``S_SLSE ~ 0.638 E_max / sqrt(T1)``.
Here we optimize the number of echoes ``K`` jointly with ``T_R`` (fewer echoes
when ``T1`` is short so the acquisition floor ``K t_E`` does not force a long
``T_R``), which is the honest "best SLSE".

*SORC.* The steady state emits ``a_ss = A M0 eta`` every cycle ``t_c`` with duty
cycle ~= 1, so ``N = T / t_c`` samples average coherently:

    S_SORC ~ a_ss / sqrt(t_c),

independent of ``T1`` in the regime ``t_c << T2e << T1`` (the Konnai steady
state). The ratio is therefore

    S_SORC / S_SLSE ~ (eta * sqrt(2) / 0.638) * sqrt(T1 / T2e),

so SORC's advantage grows as **sqrt(T1/T2e)** -- precisely because the SLSE duty
cycle collapses to ``~T2e/T1`` while SORC stays near one. The example verifies
this scaling and locates the ``T1/T2e`` crossover for crystal and powder.

Models and normalization: the SLSE echo train uses the full density-matrix
``simulate_full_slse``, which forms proper echoes and refocuses all powder
crystallites in phase -- the embedded two-level ``simulate_slse`` lets the
per-orientation coherences dephase and under-reports the powder train. The SORC
steady state is the reduced Konnai dephased-pathway model. Each scheme's signal
is normalized to *its own* model's single-pulse coil signal ``a0``; since both
``a0`` are the same physical single-pulse voltage, the per-model amplitude
constants cancel and the SORC/SLSE ratio is a true physical sensitivity ratio.
The SORC steady amplitude assumes ``tau << T2e, T1`` (negligible irreversible loss
per cycle); outside that regime SORC also degrades.

Run with ``--output nqr_slse_sorc_sensitivity.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    diagonalize_site,
    powder_average_grid,
    simulate_full_slse,
    simulate_sorc,
    single_crystal_orientation,
    slse_sequence,
    sorc_sequence,
)

X_STAR = 1.2564312086261697  # argmax of (1 - e^{-x}) / sqrt(x)


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (2.0 * np.pi * nutation_hz)


def _slse_single_pulse_ref(site, *, carrier, nutation_hz, excitation_dur, orientations):
    """Full-model single-pulse coil signal |a0|, read like the SLSE echoes.

    Uses ``simulate_full_slse`` with one "echo", a zero refocus pulse, and a tiny
    spacing, so it reads the coherence right after the excitation pulse through the
    same detection operator the echo train uses.
    """

    result = simulate_full_slse(
        site,
        nutation_hz=nutation_hz,
        excitation_duration_seconds=excitation_dur,
        refocus_duration_seconds=0.0,
        echo_spacing_seconds=excitation_dur,
        num_echoes=1,
        orientations=orientations,
        rf_frequency_hz=carrier,
        t2e_seconds=np.inf,
    )
    return float(np.abs(result.echo_amplitudes[0]))


def _sorc_single_pulse_ref(site, transition, *, nutation_hz, pulse_dur, te, orientations):
    """Reduced-model single-pulse coil signal |a0| in transition_signal units."""

    seq = slse_sequence(
        transition,
        pulse_duration_seconds=pulse_dur,
        nutation_hz=nutation_hz,
        echo_spacing_seconds=te,
        num_echoes=1,
    )
    result = _reduced_slse_single_pulse(site, seq, orientations)
    return float(np.abs(result))


def _reduced_slse_single_pulse(site, seq, orientations):
    from spin_dynamics.nqr import simulate_slse

    return simulate_slse(site, seq, orientations=orientations, t2e_seconds=np.inf).echo_amplitudes[0]


def slse_echo_energies(site, *, carrier, te, num_echoes, t2e, nutation_hz,
                       excitation_dur, refocus_dur, orientations):
    """Cumulative matched-filter energy E(K) from the full-model SLSE echo train.

    ``simulate_full_slse`` forms proper echoes (excitation, tau, refocus, tau,
    read at echo centre) so the powder average refocuses all crystallites in phase
    -- unlike the embedded two-level ``simulate_slse``, which lets the per-
    orientation coherences dephase and under-reports the powder echo train.
    """

    result = simulate_full_slse(
        site,
        nutation_hz=nutation_hz,
        excitation_duration_seconds=excitation_dur,
        refocus_duration_seconds=refocus_dur,
        echo_spacing_seconds=te,
        num_echoes=num_echoes,
        orientations=orientations,
        rf_frequency_hz=carrier,
        t2e_seconds=t2e,
    )
    amplitudes = np.abs(result.echo_amplitudes)
    return np.sqrt(np.cumsum(amplitudes**2)), amplitudes


def slse_optimal_sensitivity(energy_cumulative, te, t1):
    """Best SLSE SNR/sqrt(time) over (num echoes K, repetition time T_R)."""

    counts = np.arange(1, energy_cumulative.size + 1)
    t_acq = counts * te
    t_rep = np.maximum(t_acq, X_STAR * t1)
    recovery = (1.0 - np.exp(-t_rep / t1)) / np.sqrt(t_rep)
    sensitivity = energy_cumulative * recovery
    best = int(np.argmax(sensitivity))
    return {
        "sensitivity": float(sensitivity[best]),
        "num_echoes": int(counts[best]),
        "t_rep": float(t_rep[best]),
        "duty_cycle": float(t_acq[best] / t_rep[best]),
    }


def sorc_steady_scan(site, transition, *, tau, pulse_dur, nutation_hz, orientations, f0, phases):
    """Steady-state SORC |a_ss| vs. off-resonance phase = 2*pi*offset*tau."""

    amplitudes = np.zeros(phases.size, dtype=np.float64)
    for idx, phase in enumerate(phases):
        offset_hz = phase / (2.0 * np.pi * tau)
        seq = sorc_sequence(
            transition,
            pulse_duration_seconds=pulse_dur,
            nutation_hz=nutation_hz,
            half_spacing_seconds=tau,
            num_pulses=1,
            rf_frequency_hz=f0 + offset_hz,
        )
        result = simulate_sorc(
            site, seq, orientations=orientations, model="steady_state", t2e_seconds=np.inf
        )
        amplitudes[idx] = float(np.abs(result.signal_amplitudes[0]))
    return amplitudes


def _analyze(site, transition, args, orientations, carrier):
    """Return the sensitivity sweep and operating points for one sample type.

    SLSE is evaluated with the full density-matrix ``simulate_full_slse`` (correct
    powder refocusing); SORC with the reduced Konnai steady-state pathway. Each
    scheme's signal is normalized to *its own* model's single-pulse coil signal
    ``a0``; because both ``a0`` represent the same physical single-pulse voltage,
    the cross-model normalization cancels and the SORC/SLSE ratio is a true
    physical sensitivity ratio.
    """

    t2e = args.t2e_ms * 1e-3
    te = args.echo_spacing_us * 1e-6
    tau = args.sorc_tau_us * 1e-6
    nutation_hz = args.nutation_khz * 1e3
    excitation_dur = _pulse_duration(args.excitation_angle, nutation_hz)
    refocus_dur = _pulse_duration(args.refocus_angle, nutation_hz)
    sorc_pulse_dur = _pulse_duration(args.sorc_angle, nutation_hz)

    num_echoes = int(np.ceil(args.echo_coverage * t2e / te))
    energy_cumulative, echo_train = slse_echo_energies(
        site, carrier=carrier, te=te, num_echoes=num_echoes, t2e=t2e,
        nutation_hz=nutation_hz, excitation_dur=excitation_dur,
        refocus_dur=refocus_dur, orientations=orientations,
    )
    a0_slse = _slse_single_pulse_ref(
        site, carrier=carrier, nutation_hz=nutation_hz,
        excitation_dur=excitation_dur, orientations=orientations,
    )
    a0_sorc = _sorc_single_pulse_ref(
        site, transition, nutation_hz=nutation_hz, pulse_dur=sorc_pulse_dur,
        te=te, orientations=orientations,
    )

    phases = np.linspace(0.05 * np.pi, 2.6 * np.pi, 140)
    sorc_amplitudes = sorc_steady_scan(
        site, transition, tau=tau, pulse_dur=sorc_pulse_dur, nutation_hz=nutation_hz,
        orientations=orientations, f0=carrier, phases=phases,
    )
    a_ss = float(np.max(sorc_amplitudes))
    cycle_time = 2.0 * tau + sorc_pulse_dur
    # SORC steady signal in units of its single-pulse coil signal, per sqrt(time).
    sorc_sensitivity = (a_ss / a0_sorc) / np.sqrt(cycle_time)

    ratios = args.t1_over_t2e
    t1_values = ratios * t2e
    slse_sens = np.zeros(ratios.size)
    slse_duty = np.zeros(ratios.size)
    slse_echoes = np.zeros(ratios.size)
    energy_relative = energy_cumulative / a0_slse
    for idx, t1 in enumerate(t1_values):
        best = slse_optimal_sensitivity(energy_relative, te, t1)
        slse_sens[idx] = best["sensitivity"]
        slse_duty[idx] = best["duty_cycle"]
        slse_echoes[idx] = best["num_echoes"]

    # Both sensitivities are now in units of (single-pulse coil signal) / sqrt(s);
    # multiply by sqrt(T2e) for a dimensionless y-axis.
    scale = 1.0 / np.sqrt(t2e)
    return {
        "ratios": ratios,
        "slse_sensitivity": slse_sens / scale,
        "sorc_sensitivity": np.full(ratios.size, sorc_sensitivity / scale),
        "advantage": sorc_sensitivity / slse_sens,
        "slse_duty": slse_duty,
        "slse_echoes": slse_echoes,
        "echo_train": echo_train / a0_slse,
        "echo_times_us": (np.arange(echo_train.size) + 1) * te * 1e6,
        "phases": phases,
        "sorc_amplitudes": sorc_amplitudes / a0_sorc,
        "a0_slse": a0_slse,
        "a0_sorc": a0_sorc,
        "a_ss": a_ss,
        "num_echoes": num_echoes,
        "cycle_time": cycle_time,
    }


def _crossover(ratios, advantage):
    """Interpolate the T1/T2e where the SORC/SLSE advantage crosses unity."""

    log_r = np.log(ratios)
    log_a = np.log(advantage)
    if np.all(log_a > 0):
        return float(ratios[0])
    if np.all(log_a < 0):
        return float(ratios[-1])
    idx = int(np.argmax(np.sign(log_a[:-1]) != np.sign(log_a[1:])))
    x0, x1 = log_r[idx], log_r[idx + 1]
    y0, y1 = log_a[idx], log_a[idx + 1]
    return float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument("--quadrupole-khz", type=float, default=900.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--nutation-khz", type=float, default=20.0)
    parser.add_argument("--excitation-angle", type=float, default=90.0)
    parser.add_argument("--refocus-angle", type=float, default=180.0)
    parser.add_argument("--sorc-angle", type=float, default=90.0,
                        help="SORC hard-pulse flip angle (degrees).")
    parser.add_argument("--echo-spacing-us", type=float, default=500.0)
    parser.add_argument("--sorc-tau-us", type=float, default=250.0)
    parser.add_argument("--t2e-ms", type=float, default=10.0)
    parser.add_argument("--echo-coverage", type=float, default=5.0,
                        help="Echo train covers this many T2e.")
    parser.add_argument("--min-ratio", type=float, default=1.0)
    parser.add_argument("--max-ratio", type=float, default=1000.0)
    parser.add_argument("--num-ratios", type=int, default=28)
    parser.add_argument("--n-theta", type=int, default=10)
    parser.add_argument("--n-phi", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Single-crystal azimuth (rad); avoid nodal orientations.")
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Single-crystal polar angle (rad).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.t1_over_t2e = np.geomspace(args.min_ratio, args.max_ratio, args.num_ratios)
    return args


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
    )
    carrier = diagonalize_site(site).transition(args.transition).frequency_hz
    crystal = single_crystal_orientation(alpha=args.alpha, beta=args.beta)
    powder = powder_average_grid(args.n_theta, args.n_phi)

    results = {
        "crystal": _analyze(site, args.transition, args, crystal, carrier),
        "powder": _analyze(site, args.transition, args, powder, carrier),
    }
    _plot(plt, results, args)
    _print_summary(results)


def _plot(plt, results, args):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    colors = {"crystal": "C0", "powder": "C1"}

    # (0,0) SLSE echo train envelope (full-model powder refocuses in phase).
    for name, res in results.items():
        axes[0, 0].plot(res["echo_times_us"] / 1e3, res["echo_train"],
                        color=colors[name], lw=0.9, label=f"{name}")
    axes[0, 0].set_title("SLSE Echo Train (full model, T2e envelope)")
    axes[0, 0].set_xlabel("echo time (ms)")
    axes[0, 0].set_ylabel("|echo| / single-pulse a0")
    axes[0, 0].legend(fontsize=8)

    # (0,1) SORC steady-state off-resonance response (operating point).
    for name, res in results.items():
        axes[0, 1].plot(res["phases"] / np.pi, res["sorc_amplitudes"],
                        color=colors[name], lw=1.0, label=name)
    axes[0, 1].set_title("SORC Steady-State vs. Offset")
    axes[0, 1].set_xlabel(r"off-resonance phase $2\pi\,\delta\,\tau$ / $\pi$")
    axes[0, 1].set_ylabel(r"$|a_{ss}|$ / single-pulse a0")
    axes[0, 1].legend(fontsize=8)

    # (0,2) Sensitivity per sqrt(time), single crystal.
    res = results["crystal"]
    axes[0, 2].loglog(res["ratios"], res["slse_sensitivity"], "o-", color="C3", ms=3, label="SLSE")
    axes[0, 2].loglog(res["ratios"], res["sorc_sensitivity"], "s-", color="C2", ms=3, label="SORC")
    axes[0, 2].set_title("Sensitivity / sqrt(time): Single Crystal")
    axes[0, 2].set_xlabel(r"$T_1 / T_{2e}$")
    axes[0, 2].set_ylabel("SNR per sqrt(time) [norm.]")
    axes[0, 2].legend(fontsize=8)

    # (1,0) Sensitivity per sqrt(time), powder.
    res = results["powder"]
    axes[1, 0].loglog(res["ratios"], res["slse_sensitivity"], "o-", color="C3", ms=3, label="SLSE")
    axes[1, 0].loglog(res["ratios"], res["sorc_sensitivity"], "s-", color="C2", ms=3, label="SORC")
    axes[1, 0].set_title("Sensitivity / sqrt(time): Powder")
    axes[1, 0].set_xlabel(r"$T_1 / T_{2e}$")
    axes[1, 0].set_ylabel("SNR per sqrt(time) [norm.]")
    axes[1, 0].legend(fontsize=8)

    # (1,1) SORC/SLSE advantage with the sqrt(T1/T2e) law.
    for name, res in results.items():
        axes[1, 1].loglog(res["ratios"], res["advantage"], "o-", color=colors[name], ms=3, label=name)
    ref_ratios = results["crystal"]["ratios"]
    anchor = results["crystal"]["advantage"][-1] / np.sqrt(ref_ratios[-1])
    axes[1, 1].loglog(ref_ratios, anchor * np.sqrt(ref_ratios), "k--", lw=1.0,
                      label=r"$\propto\sqrt{T_1/T_{2e}}$")
    axes[1, 1].axhline(1.0, color="0.5", lw=0.8, ls=":")
    axes[1, 1].set_title("SORC / SLSE Advantage")
    axes[1, 1].set_xlabel(r"$T_1 / T_{2e}$")
    axes[1, 1].set_ylabel("SNR-per-time ratio")
    axes[1, 1].legend(fontsize=8)

    # (1,2) SLSE duty cycle collapse (the mechanism).
    for name, res in results.items():
        axes[1, 2].loglog(res["ratios"], res["slse_duty"], color=colors[name], lw=1.2, label=f"{name} duty")
    axes[1, 2].loglog(results["crystal"]["ratios"],
                      1.0 / results["crystal"]["ratios"], "k:", lw=1.0, label=r"$\propto T_{2e}/T_1$")
    axes[1, 2].set_title("SLSE Duty Cycle (acq / T_R)")
    axes[1, 2].set_xlabel(r"$T_1 / T_{2e}$")
    axes[1, 2].set_ylabel("optimal duty cycle")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("SLSE vs. Steady-State SORC: NQR Sensitivity per Unit Time")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(results):
    print("SLSE vs. steady-state SORC sensitivity per unit time")
    for name, res in results.items():
        ratios = res["ratios"]
        advantage = res["advantage"]
        crossover = _crossover(ratios, advantage)
        # Asymptotic log-log slope of the advantage over the top decade.
        top = ratios >= ratios[-1] / 10.0
        slope = float(np.polyfit(np.log(ratios[top]), np.log(advantage[top]), 1)[0])
        print(
            f"  {name:8s}: a0(SLSE full)={res['a0_slse']:.3f}, "
            f"a0(SORC reduced)={res['a0_sorc']:.3f}, "
            f"SORC eta={res['a_ss'] / res['a0_sorc']:.2f}, echoes<= {res['num_echoes']}"
        )
        print(
            f"            SORC/SLSE advantage: {advantage[0]:.2f}x at T1/T2e={ratios[0]:.0f} "
            f"-> {advantage[-1]:.1f}x at T1/T2e={ratios[-1]:.0f}"
        )
        print(
            f"            crossover T1/T2e (SORC overtakes SLSE): {crossover:.2f}; "
            f"asymptotic slope={slope:.2f} (expected 0.5)"
        )


if __name__ == "__main__":
    main()
