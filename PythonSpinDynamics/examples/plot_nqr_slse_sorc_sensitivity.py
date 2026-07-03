r"""SLSE vs. steady-state SORC NQR sensitivity per unit time, done honestly.

This example compares the SNR per unit time of the two multiple-pulse ^14N NQR
detection schemes -- spin-lock spin-echo (SLSE) and the steady-state strong
off-resonance comb (SORC) -- for a *specific material* (NaNO2) whose relaxation
is set by a microscopic dipolar bath, so the answer is grounded rather than
parametric.

Why it is built this way (each choice fixes a way of getting the wrong answer):

* **Three time constants, and none is inserted by hand.** Solids have a fast
  *reversible* dephasing (T2\*, the rigid-lattice static dipolar local field,
  refocused by the pulse train), a long spin-locked decay (T2e, a T1rho, that
  survives the train), and a long T1. T2e and T1 *emerge* from a microscopic
  Redfield dipolar-relaxation model built from NaNO2's actual neighbour spins
  (``RedfieldDipolarRelaxationModel.from_dipolar_sources``); T2\* is the Van Vleck
  rigid-lattice second moment of the *same* neighbour list (weak here -- ^14N has
  a small gyromagnetic ratio, so T2\* ~ 1 ms, set mostly by the ^23Na neighbours,
  not the "tens of us" of a proton-rich solid). The correlation time ``tau_c`` is
  the physical, temperature-controlled knob; sweeping it walks the material
  through its relaxation regimes (T1/T2e ~ 1 warm to ~30 cold) while T2\* stays
  fixed. A different sample or temperature is a separate calculation.
* **One common normalization.** Both sequences are propagated in the same full
  density-matrix framework from the same equilibrium ``rho_eq`` and read with the
  same detection operator, so their signals share one scale -- no per-sequence
  normalization. The SORC steady state is the affine fixed point of its cycle
  with an equilibrium-target source ``(I - M) rho_eq`` (a steady-state sequence
  has zero signal without T1 replenishing the polarized equilibrium).
* **Matched-filter SNR.** Sensitivity uses the time-domain energy
  ``sqrt(integral |s(t)|^2 dt)`` in each acquisition window, not a peak.
* **Both sequences fully optimized.** Flip angle, RF offset, and pulse spacing
  are optimized for *each* scheme at *each* ``tau_c`` (the small-tip, high-
  retention SORC optimum is what keeps its steady state alive at long T1).

Result: once all of the above are correct, SLSE and SORC are *comparable*. On a
single crystal they stay within ~15% of each other across the whole T1/T2e range
(SORC marginally ahead when warm, SLSE marginally ahead when cold) -- neither
grows a running advantage. The naive "SORC wins as sqrt(T1/T2e)" is an artifact
of a T1-independent steady-state model; "SLSE wins increasingly" is an artifact of
under-optimizing the SORC flip angle. The one robust *asymmetry* shows up in the
**powder**: SORC pulls modestly ahead (~1.1x) because its small-tip steady state
is far more uniform across the crystallite orientation distribution, whereas the
SLSE 90/180 echo has a strongly orientation-dependent refocusing efficiency (some
crystallites sit near a nutation null), so it loses more to powder averaging.

A second asymmetry is in **average RF power**. Both schemes drive the same coil
at the same nutation frequency, so peak power is identical and the average-power
ratio is just a duty-cycle ratio. At the optima SORC actually draws *less* average
power (~0.2-0.3x SLSE): its small ~15 deg tip makes each pulse short even though it
fires continuously, whereas the SLSE matched filter wants wide ~70 deg pulses at
the shortest allowed spacing (a high in-train duty). So the naive "continuous SORC
must dissipate more power" is also wrong here -- though note SLSE could trade to a
longer echo spacing for lower power at little SNR cost, so this is an operating-
point statement, not a fundamental bound.

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
    simulate_full_echo,
    simulate_full_slse,
    single_crystal_orientation,
)
from spin_dynamics.nqr.full_dynamics import (  # noqa: E402
    detection_operator,
    pulse_hamiltonian,
    static_hamiltonian_rotating,
)
from spin_dynamics.nqr.simulation import equilibrium_density  # noqa: E402
from spin_dynamics.relaxation import (  # noqa: E402
    NQRRelaxationModel,
    RedfieldDipolarRelaxationModel,
    RigidSolidMotionalAveraging,
    dipolar_coupling_hz,
    liouville_superoperator,
    matrix_exponential,
)
from plot_redfield_nano2_slse import (  # noqa: E402
    _load_nano2_parameters,
    _nano2_dipolar_sources,
)

_NO_RELAX = NQRRelaxationModel(t1_seconds=np.inf, t2_seconds=np.inf)
_X_STAR = 1.2564312086261697  # argmax of (1 - e^{-x}) / sqrt(x)


def _trapz(y, x):
    return float(np.trapezoid(np.abs(y) ** 2, x))


def static_second_moment(sources):
    """Rigid-lattice Van Vleck dipolar second moment M2 (rad^2/s^2).

    This is the *reversible* static dephasing that both pulse trains refocus. It
    is computed from the same neighbour list that feeds the Redfield model, using
    the standard like/unlike Van Vleck coefficients (3/4 and 1/3) with the
    powder-averaged angular factor <(1 - 3 cos^2 theta)^2> = 4/5, so it is an
    order-of-magnitude, orientation-independent estimate rather than a full NQR
    line-shape calculation. ``T2* = sqrt(2 / M2)`` is its Gaussian 1/e time.
    """

    m2 = 0.0
    for src in sources:
        distance = float(np.linalg.norm(np.asarray(src.vector_angstrom, dtype=np.float64)))
        coupling = dipolar_coupling_hz(
            distance, gamma_a_hz_per_t=src.gamma_target_hz_per_t,
            gamma_b_hz_per_t=src.gamma_bath_hz_per_t)
        spin = float(src.bath_spin)
        like = abs(src.gamma_bath_hz_per_t - src.gamma_target_hz_per_t) < 1.0
        coeff = (3.0 / 5.0 if like else 4.0 / 15.0) * spin * (spin + 1.0)
        m2 += coeff * (2.0 * np.pi * coupling) ** 2
    return m2


def _static_dephasing(times, center, m2):
    """Gaussian reversible-dephasing envelope refocused at ``center``."""

    return np.exp(-0.5 * m2 * (np.asarray(times, dtype=np.float64) - center) ** 2)


def _build_material():
    params = _load_nano2_parameters()
    site = QuadrupolarSite(
        spin=1.0, isotope="14N", label="NaNO2 N1",
        quadrupole_frequency_hz=params.nu_q_hz, eta=params.eta, gamma_hz_per_t=3.0766e6,
    )
    sources, _, _ = _nano2_dipolar_sources(radius_angstrom=5.0, max_neighbors=10)
    eig = diagonalize_site(site)
    carrier = eig.transition("x").frequency_hz
    return site, eig, carrier, sources


def _relaxation(sources, tau_c):
    return RedfieldDipolarRelaxationModel.from_dipolar_sources(
        1.0, sources, motion=RigidSolidMotionalAveraging(tau_c))


def _fit_time_constant(times, magnitudes):
    times = np.asarray(times, dtype=np.float64)
    magnitudes = np.abs(np.asarray(magnitudes, dtype=np.complex128))
    ok = magnitudes > 0
    if np.count_nonzero(ok) < 2:
        return np.inf
    slope = np.polyfit(times[ok] - times[ok][0], np.log(magnitudes[ok]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else np.inf


def emergent_times(site, eig, carrier, relax, nut, orientations, m2):
    """Return (T1, T2star, T2e): T1/T2e from Redfield, T2star from static moment."""

    deq = equilibrium_density(eig.levels_hz)
    generator = liouville_superoperator(static_hamiltonian_rotating(eig, carrier), relax)
    vec = deq.reshape(-1, order="F")
    ts = np.geomspace(3e-4, 3e-1, 12)
    p0 = float(np.real(deq[0, 0]))
    pol = np.array([
        np.real((matrix_exponential(generator, t) @ vec).reshape(3, 3, order="F")[0, 0])
        for t in ts
    ])
    t1 = _fit_time_constant(ts, pol / p0) if p0 != 0 else np.inf

    # Reversible FID decay is dominated by the fast static dephasing T2* (Gaussian
    # 1/e), which is temperature-independent (rigid lattice) unlike T1 and T2e.
    t2star = np.sqrt(2.0 / m2) if m2 > 0 else np.inf

    dur = np.deg2rad(90.0) / (2.0 * np.pi * nut)
    train = simulate_full_slse(site, nutation_hz=nut, excitation_duration_seconds=dur,
                               refocus_duration_seconds=dur, echo_spacing_seconds=200e-6,
                               num_echoes=60, orientations=orientations,
                               rf_frequency_hz=carrier, relaxation=relax)
    t2e = _fit_time_constant(train.echo_times, train.echo_amplitudes)
    return t1, t2star, t2e


# --------------------------------------------------------------------------- SLSE

def slse_locked_t2e(site, eig, carrier, relax, nut, te, orientations, *, num_echoes=40):
    """Spin-locked T2e at the given echo spacing, measured at the canonical 90 lock.

    T2e is a property of the *locked* state, so it is measured once per spacing at
    a well-locked 90 flip (where the train is monotone) rather than refit at every
    trial flip -- an exponential fit to the near-null, non-monotone trains of
    off-optimal flips is what made the raw nutation curve spuriously spiky. The
    flip dependence is carried entirely by the smooth per-echo energy ``q_echo``.
    """

    dur = np.deg2rad(90.0) / (2.0 * np.pi * nut)
    train = simulate_full_slse(site, nutation_hz=nut, excitation_duration_seconds=dur,
                               refocus_duration_seconds=dur, echo_spacing_seconds=te,
                               num_echoes=num_echoes, orientations=orientations,
                               rf_frequency_hz=carrier, relaxation=relax)
    return _fit_time_constant(train.echo_times, train.echo_amplitudes)


def slse_matched_energy(site, eig, carrier, relax, nut, *, flip, offset, te,
                        orientations, m2):
    """Return (per-echo matched-filter energy q_echo, echo shape) in common units."""

    dur = np.deg2rad(flip) / (2.0 * np.pi * nut)
    rf = carrier + offset
    tw = np.linspace(0.0, te, 32)
    # Powder echo shape: coherent orientation sum of single-echo signals.
    shape = np.zeros(tw.size, dtype=np.complex128)
    total = 0.0
    for sample in orientations:
        echo = simulate_full_echo(site, nutation_hz=nut, excitation_duration_seconds=dur,
                                  refocus_duration_seconds=dur, echo_spacing_seconds=te,
                                  times_seconds=tw, rf_frequency_hz=rf,
                                  b1_direction_pas=sample.b1_direction_pas, relaxation=relax)
        shape = shape + sample.weight * echo.signal
        total += sample.weight
    shape = shape / (total or 1.0)
    # The 180 refocuses the reversible static dephasing at the echo centre te/2.
    shape = shape * _static_dephasing(tw, 0.5 * te, m2)
    return _trapz(shape, tw), shape, tw


def slse_sensitivity(q_echo, t2e, te, t1, *, operating_point=False):
    """Best matched-filter SNR / sqrt(time) over echo count K and repetition T_R.

    With ``operating_point=True`` also return the optimal ``(K, T_R)`` so the
    average RF duty cycle at the optimum can be evaluated.
    """

    if not np.isfinite(t2e) or t2e <= 0 or q_echo <= 0:
        return (0.0, 0, np.inf) if operating_point else 0.0
    counts = np.arange(1, 4000)
    envelope = (1.0 - np.exp(-2.0 * counts * te / t2e)) / (1.0 - np.exp(-2.0 * te / t2e))
    e_train = np.sqrt(q_echo * envelope)
    t_acq = counts * te
    t_rep = np.maximum(t_acq, _X_STAR * t1)
    recovery = (1.0 - np.exp(-t_rep / t1)) / np.sqrt(t_rep)
    score = e_train * recovery
    idx = int(np.argmax(score))
    if operating_point:
        return float(score[idx]), int(counts[idx]), float(t_rep[idx])
    return float(score[idx])


def slse_optimize(site, eig, carrier, relax, nut, t1, orientations, m2, *,
                  flips, offsets, spacings):
    """Maximize SLSE sensitivity over (flip, offset, echo spacing)."""

    best = {"sensitivity": 0.0}
    for te in spacings:
        t2e = slse_locked_t2e(site, eig, carrier, relax, nut, te, orientations)
        for flip in flips:
            for offset in offsets:
                q_echo, _, _ = slse_matched_energy(
                    site, eig, carrier, relax, nut, flip=flip, offset=offset, te=te,
                    orientations=orientations, m2=m2)
                sens, k_opt, t_rep = slse_sensitivity(q_echo, t2e, te, t1, operating_point=True)
                if sens > best["sensitivity"]:
                    best = {"sensitivity": sens, "flip": flip, "offset": offset,
                            "te": te, "t2e": t2e, "num_echoes": k_opt, "t_rep": t_rep}
    return best


# --------------------------------------------------------------------------- SORC

def sorc_window(site, eig, carrier, relax, nut, *, flip, offset, tau, orientations, m2):
    """Affine steady-state SORC acquisition window (common signal units)."""

    tp = np.deg2rad(flip) / (2.0 * np.pi * nut)
    rf = carrier + offset
    dim = eig.levels_hz.size
    identity = np.eye(dim * dim)
    deq = equilibrium_density(eig.levels_hz).reshape(-1, order="F")
    free_h = static_hamiltonian_rotating(eig, rf)
    generator = liouville_superoperator(free_h, relax)
    free2 = matrix_exponential(generator, 2.0 * tau)
    source = (identity - free2) @ deq
    lam, vecs = np.linalg.eig(generator)
    inv_vecs = np.linalg.inv(vecs)
    tw = np.linspace(0.0, 2.0 * tau, 80)
    signal = np.zeros(tw.size, dtype=np.complex128)
    total = 0.0
    for sample in orientations:
        pulse_h = pulse_hamiltonian(eig, nutation_hz=nut, rf_frequency_hz=rf, phase=0.0,
                                    b1_direction_pas=sample.b1_direction_pas)
        pulse = matrix_exponential(liouville_superoperator(pulse_h, _NO_RELAX), tp)
        det = detection_operator(eig, rf, sample.b1_direction_pas)
        steady = np.linalg.solve(identity - free2 @ pulse, source)
        rho0 = pulse @ steady
        local = np.empty(tw.size, dtype=np.complex128)
        for i, t in enumerate(tw):
            prop = (vecs * np.exp(lam * t)) @ inv_vecs
            rho = prop @ rho0 + (identity - prop) @ deq
            local[i] = np.trace(rho.reshape(dim, dim, order="F") @ det)
        signal = signal + sample.weight * local
        total += sample.weight
    signal = signal / (total or 1.0)
    # SSFP-like refocusing: the reversible static dephasing forms an FID after each
    # pulse and a multiple-pulse echo before the next, so the envelope is a tent
    # pinned near 1 at both window edges (distance to the nearest pulse). At the
    # SORC optimum the comb spacing is << T2*, so this envelope is ~1 and the exact
    # shape (tent vs. plain FID) is immaterial -- the ratio is robust to it.
    signal = signal * _static_dephasing(np.minimum(tw, 2.0 * tau - tw), 0.0, m2)
    return signal, tw, 2.0 * tau + tp


def sorc_sensitivity(site, eig, carrier, relax, nut, *, flip, offset, tau, orientations, m2):
    signal, tw, cycle = sorc_window(site, eig, carrier, relax, nut, flip=flip,
                                    offset=offset, tau=tau, orientations=orientations, m2=m2)
    return np.sqrt(_trapz(signal, tw) / cycle)


def sorc_optimize(site, eig, carrier, relax, nut, orientations, m2, *, flips, phases, taus):
    """Maximize SORC sensitivity over (flip, offset, half spacing)."""

    best = {"sensitivity": 0.0}
    for tau in taus:
        for flip in flips:
            for phase in phases:
                offset = phase / (2.0 * np.pi * tau)
                sens = sorc_sensitivity(site, eig, carrier, relax, nut, flip=flip,
                                        offset=offset, tau=tau, orientations=orientations, m2=m2)
                if sens > best["sensitivity"]:
                    best = {"sensitivity": sens, "flip": flip, "offset": offset,
                            "tau": tau, "phase": phase}
    return best


# ------------------------------------------------------------------- RF power

def _pulse_seconds(flip_deg, nut):
    return np.deg2rad(flip_deg) / (2.0 * np.pi * nut)


def average_rf_power(slse_opt, sorc_opt, nut):
    """Average RF power (relative to peak) at each optimum, and the SORC/SLSE ratio.

    Both schemes drive the same coil at the same nutation frequency, so the peak
    (in-pulse) power is identical and the *average* RF power is just peak x duty
    cycle. The comparison is therefore a duty-cycle ratio. Two duties matter:

    * ``duty_active`` -- the in-train pulse density, ``t_p / cycle``. SORC's small
      tip makes each pulse short, but it fires every ``2 tau``; SLSE fires one
      wider refocusing pulse every ``t_E``.
    * ``duty_avg`` -- the experiment-averaged duty. SORC is a *continuous* steady
      state (no dead time), so its average equals its active duty. SLSE spends
      most of a repetition ``T_R ~ 1.26 T_1`` recovering with the transmitter
      **off**, so its ``(K+1)`` pulses are amortized over ``T_R`` and its average
      duty collapses far below its active duty.
    """

    tp_slse = _pulse_seconds(slse_opt["flip"], nut)
    tp_sorc = _pulse_seconds(sorc_opt["flip"], nut)
    cycle_sorc = 2.0 * sorc_opt["tau"] + tp_sorc

    active_slse = tp_slse / slse_opt["te"]
    active_sorc = tp_sorc / cycle_sorc
    # SLSE: one excitation + K refocusing pulses amortized over the repetition T_R.
    avg_slse = (slse_opt.get("num_echoes", 0) + 1) * tp_slse / slse_opt.get("t_rep", np.inf)
    avg_sorc = active_sorc  # continuous steady state, no recovery dead time

    return {
        "active_slse": active_slse, "active_sorc": active_sorc,
        "avg_slse": avg_slse, "avg_sorc": avg_sorc,
        "ratio_active": active_sorc / active_slse if active_slse > 0 else np.inf,
        "ratio_avg": avg_sorc / avg_slse if avg_slse > 0 else np.inf,
    }


# --------------------------------------------------------------------------- run

def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nutation-khz", type=float, default=25.0)
    parser.add_argument("--tau-c-us", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                        help="Correlation times to sweep (temperature knob).")
    parser.add_argument("--ref-tau-c-us", type=float, default=8.0,
                        help="tau_c for the nutation/offset/time-domain/powder panels.")
    parser.add_argument("--powder-n-theta", type=int, default=6)
    parser.add_argument("--powder-n-phi", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sweep(site, eig, carrier, sources, nut, tau_list, crystal, m2):
    flips_slse = np.array([70.0, 90.0, 105.0, 120.0, 180.0])
    offs = np.array([0.0, 2.0e3])
    spac = np.geomspace(200e-6, 1400e-6, 4)
    flips_sorc = np.array([8.0, 15.0, 30.0, 60.0, 120.0])
    phases = np.linspace(0.2 * np.pi, 2.2 * np.pi, 10)
    taus = np.geomspace(130e-6, 650e-6, 4)

    rows = []
    for tau_c in tau_list:
        relax = _relaxation(sources, tau_c)
        t1, t2, t2e = emergent_times(site, eig, carrier, relax, nut, crystal, m2)
        slse = slse_optimize(site, eig, carrier, relax, nut, t1, crystal, m2,
                             flips=flips_slse, offsets=offs, spacings=spac)
        sorc = sorc_optimize(site, eig, carrier, relax, nut, crystal, m2,
                            flips=flips_sorc, phases=phases, taus=taus)
        power = average_rf_power(slse, sorc, nut)
        rows.append({"tau_c": tau_c, "t1": t1, "t2": t2, "t2e": t2e,
                     "slse": slse, "sorc": sorc, "power": power})
        print(f"tau_c={tau_c*1e6:5.1f}us T1={t1*1e3:7.1f}ms T2e={t2e*1e3:6.1f}ms "
              f"(T1/T2e={t1/t2e:5.1f}) SLSE={slse['sensitivity']:.3e} "
              f"SORC={sorc['sensitivity']:.3e} SORC/SLSE={sorc['sensitivity']/slse['sensitivity']:.3f} "
              f"| RF duty avg SLSE={power['avg_slse']*100:.2f}% SORC={power['avg_sorc']*100:.2f}% "
              f"(SORC/SLSE={power['ratio_avg']:.1f}x)")
    return rows


def _diagnostic_curves(site, eig, carrier, relax, nut, t1, crystal, slse_opt, sorc_opt, m2):
    """SNR vs flip and vs offset for both schemes at the reference tau_c."""

    t2e = slse_locked_t2e(site, eig, carrier, relax, nut, slse_opt["te"], crystal)

    flips = np.linspace(10.0, 200.0, 18)
    slse_flip = np.array([
        slse_sensitivity(slse_matched_energy(site, eig, carrier, relax, nut, flip=f,
                         offset=slse_opt["offset"], te=slse_opt["te"], orientations=crystal,
                         m2=m2)[0], t2e, slse_opt["te"], t1) for f in flips])
    sorc_flip = np.array([
        sorc_sensitivity(site, eig, carrier, relax, nut, flip=f, offset=sorc_opt["offset"],
                         tau=sorc_opt["tau"], orientations=crystal, m2=m2) for f in flips])

    offsets = np.linspace(-6e3, 6e3, 19)
    slse_off = np.array([
        slse_sensitivity(slse_matched_energy(site, eig, carrier, relax, nut,
                         flip=slse_opt["flip"], offset=o, te=slse_opt["te"], orientations=crystal,
                         m2=m2)[0], t2e, slse_opt["te"], t1) for o in offsets])
    sorc_off = np.array([
        sorc_sensitivity(site, eig, carrier, relax, nut, flip=sorc_opt["flip"], offset=o,
                         tau=sorc_opt["tau"], orientations=crystal, m2=m2) for o in offsets])
    return {"flips": flips, "slse_flip": slse_flip, "sorc_flip": sorc_flip,
            "offsets": offsets, "slse_off": slse_off, "sorc_off": sorc_off}


def main():
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    site, eig, carrier, sources = _build_material()
    nut = args.nutation_khz * 1e3
    crystal = single_crystal_orientation(alpha=0.3, beta=0.7)
    powder = powder_average_grid(args.powder_n_theta, args.powder_n_phi)

    m2 = static_second_moment(sources)
    print(f"NaNO2 14N: transition x = {carrier/1e6:.4f} MHz, nutation {nut/1e3:.0f} kHz, "
          f"static T2* = {np.sqrt(2.0/m2)*1e6:.0f} us ({np.sqrt(m2)/(2*np.pi):.0f} Hz)")
    rows = _sweep(site, eig, carrier, sources, nut, [t * 1e-6 for t in args.tau_c_us], crystal, m2)

    # Reference tau_c: diagnostic curves, time-domain windows, powder point.
    ref = min(rows, key=lambda r: abs(r["tau_c"] - args.ref_tau_c_us * 1e-6))
    relax = _relaxation(sources, ref["tau_c"])
    curves = _diagnostic_curves(site, eig, carrier, relax, nut, ref["t1"], crystal,
                                ref["slse"], ref["sorc"], m2)
    _, slse_shape, slse_tw = slse_matched_energy(
        site, eig, carrier, relax, nut, flip=ref["slse"]["flip"], offset=ref["slse"]["offset"],
        te=ref["slse"]["te"], orientations=crystal, m2=m2)
    sorc_shape, sorc_tw, _ = sorc_window(site, eig, carrier, relax, nut,
                                         flip=ref["sorc"]["flip"], offset=ref["sorc"]["offset"],
                                         tau=ref["sorc"]["tau"], orientations=crystal, m2=m2)

    print("Powder at reference tau_c...")
    slse_p = slse_optimize(site, eig, carrier, relax, nut, ref["t1"], powder, m2,
                          flips=np.array([70.0, 90.0, 105.0, 120.0, 150.0]),
                          offsets=np.array([0.0, 2e3]), spacings=np.geomspace(200e-6, 1400e-6, 5))
    sorc_p = sorc_optimize(site, eig, carrier, relax, nut, powder, m2,
                          flips=np.array([10.0, 20.0, 40.0, 90.0]),
                          phases=np.linspace(0.2 * np.pi, 2.2 * np.pi, 10),
                          taus=np.geomspace(150e-6, 600e-6, 4))
    print(f"  powder SLSE={slse_p['sensitivity']:.3e} SORC={sorc_p['sensitivity']:.3e} "
          f"SORC/SLSE={sorc_p['sensitivity']/slse_p['sensitivity']:.3f}")

    _plot(plt, args, rows, ref, curves, slse_shape, slse_tw, sorc_shape, sorc_tw, slse_p, sorc_p)


def _plot(plt, args, rows, ref, curves, slse_shape, slse_tw, sorc_shape, sorc_tw, slse_p, sorc_p):
    tau_us = np.array([r["tau_c"] for r in rows]) * 1e6
    t1 = np.array([r["t1"] for r in rows])
    t2 = np.array([r["t2"] for r in rows])
    t2e = np.array([r["t2e"] for r in rows])
    ratio_t = t1 / t2e
    slse_s = np.array([r["slse"]["sensitivity"] for r in rows])
    sorc_s = np.array([r["sorc"]["sensitivity"] for r in rows])
    power_avg = np.array([r["power"]["ratio_avg"] for r in rows])
    order = np.argsort(ratio_t)

    fig, ax = plt.subplots(3, 3, figsize=(15.0, 12.0), constrained_layout=True)

    ax[0, 0].loglog(tau_us, t1 * 1e3, "o-", label="T1")
    ax[0, 0].loglog(tau_us, t2 * 1e3, "s-", label="T2* (static, FID)")
    ax[0, 0].loglog(tau_us, t2e * 1e3, "^-", label="T2e (SLSE)")
    ax[0, 0].set_xlabel(r"correlation time $\tau_c$ (us)  [~ temperature]")
    ax[0, 0].set_ylabel("emergent time (ms)")
    ax[0, 0].set_title("Emergent relaxation (NaNO2 Redfield)")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].loglog(ratio_t[order], slse_s[order], "o-", color="C3", label="SLSE")
    ax[0, 1].loglog(ratio_t[order], sorc_s[order], "s-", color="C2", label="SORC")
    ax[0, 1].set_xlabel(r"$T_1 / T_{2e}$")
    ax[0, 1].set_ylabel("matched-filter SNR / sqrt(time)")
    ax[0, 1].set_title("Optimized sensitivity (single crystal)")
    ax[0, 1].legend(fontsize=8)

    ax[0, 2].semilogx(ratio_t[order], (sorc_s / slse_s)[order], "o-", color="C0",
                      label="sensitivity")
    ax[0, 2].axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax[0, 2].set_xlabel(r"$T_1 / T_{2e}$")
    ax[0, 2].set_ylabel("SORC / SLSE  sensitivity", color="C0")
    ax[0, 2].tick_params(axis="y", labelcolor="C0")
    ax[0, 2].set_title("Sensitivity vs. RF-power cost (crystal)")
    axp = ax[0, 2].twinx()
    axp.semilogx(ratio_t[order], power_avg[order], "s--", color="C4", label="avg RF power")
    axp.set_ylabel("SORC / SLSE  avg RF power", color="C4")
    axp.tick_params(axis="y", labelcolor="C4")
    axp.set_ylim(bottom=0.0)

    ax[1, 0].plot(curves["flips"], curves["slse_flip"], color="C3", label="SLSE")
    ax[1, 0].plot(curves["flips"], curves["sorc_flip"], color="C2", label="SORC")
    ax[1, 0].axvline(ref["slse"]["flip"], color="C3", ls=":", lw=0.8)
    ax[1, 0].axvline(ref["sorc"]["flip"], color="C2", ls=":", lw=0.8)
    ax[1, 0].set_xlabel("pulse flip angle (deg)")
    ax[1, 0].set_ylabel("SNR / sqrt(time)")
    ax[1, 0].set_title(f"Nutation curve (tau_c={ref['tau_c']*1e6:.0f}us)")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(curves["offsets"] / 1e3, curves["slse_off"], color="C3", label="SLSE")
    ax[1, 1].plot(curves["offsets"] / 1e3, curves["sorc_off"], color="C2", label="SORC")
    ax[1, 1].axvline(ref["slse"]["offset"] / 1e3, color="C3", ls=":", lw=0.8)
    ax[1, 1].axvline(ref["sorc"]["offset"] / 1e3, color="C2", ls=":", lw=0.8)
    ax[1, 1].set_xlabel("RF offset (kHz)")
    ax[1, 1].set_ylabel("SNR / sqrt(time)")
    ax[1, 1].set_title("Offset dependence")
    ax[1, 1].legend(fontsize=8)

    labels = ["SLSE", "SORC"]
    xtal_vals = [ref["slse"]["sensitivity"], ref["sorc"]["sensitivity"]]
    powder_vals = [slse_p["sensitivity"], sorc_p["sensitivity"]]
    xloc = np.arange(2)
    ax[1, 2].bar(xloc - 0.18, xtal_vals, width=0.35, color=["C3", "C2"], label="single crystal")
    ax[1, 2].bar(xloc + 0.18, powder_vals, width=0.35, color=["C3", "C2"], alpha=0.5, label="powder")
    ax[1, 2].set_xticks(xloc, labels)
    ax[1, 2].set_ylabel("SNR / sqrt(time)")
    ax[1, 2].set_title(f"Crystal vs powder (tau_c={ref['tau_c']*1e6:.0f}us)")
    ax[1, 2].legend(fontsize=8)

    ax[2, 0].plot(slse_tw * 1e6, np.real(slse_shape), color="C3", label="Re")
    ax[2, 0].plot(slse_tw * 1e6, np.abs(slse_shape), "k--", lw=0.8, label="|s|")
    ax[2, 0].set_xlabel("time in window (us)")
    ax[2, 0].set_ylabel("signal")
    ax[2, 0].set_title(f"SLSE window (echo, te={ref['slse']['te']*1e6:.0f}us << T2*)")
    ax[2, 0].legend(fontsize=8)

    ax[2, 1].plot(sorc_tw * 1e6, np.real(sorc_shape), color="C2", label="Re")
    ax[2, 1].plot(sorc_tw * 1e6, np.abs(sorc_shape), "k--", lw=0.8, label="|s|")
    ax[2, 1].set_xlabel("time in window (us)")
    ax[2, 1].set_ylabel("signal")
    ax[2, 1].set_title("SORC acquisition window (steady state)")
    ax[2, 1].legend(fontsize=8)

    ax[2, 2].axis("off")
    ratio_x = ref["sorc"]["sensitivity"] / ref["slse"]["sensitivity"]
    ratio_p = sorc_p["sensitivity"] / slse_p["sensitivity"]
    pw = ref["power"]
    lines = [
        "Honest comparison (NaNO2 14N):",
        "* T1, T2e emerge from the microscopic",
        "  dipolar Redfield model (not inserted);",
        "  T2* from the Van Vleck static moment.",
        "* both schemes optimized over flip,",
        "  offset, spacing; matched filter;",
        "  one common normalization.",
        "",
        f"At T1/T2e={ref['t1']/ref['t2e']:.0f}:",
        f"  crystal SORC/SLSE = {ratio_x:.2f}",
        f"  powder  SORC/SLSE = {ratio_p:.2f}",
        f"SORC opt: flip={ref['sorc']['flip']:.0f}deg, "
        f"2tau={2*ref['sorc']['tau']*1e6:.0f}us",
        f"SLSE opt: flip={ref['slse']['flip']:.0f}deg, "
        f"te={ref['slse']['te']*1e6:.0f}us",
        "",
        "Avg RF power (same coil & B1):",
        f"  SLSE {pw['avg_slse']*100:.2f}%  SORC {pw['avg_sorc']*100:.2f}% duty",
        f"  => SORC draws {pw['ratio_avg']:.2f}x SLSE's avg power",
        "     (small tip beats wide short-te pulses).",
        "",
        "=> comparable SNR on a crystal; SORC is",
        "   more powder-robust AND lower-power,",
        "   at the cost of continuous operation.",
    ]
    ax[2, 2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
                  family="monospace", transform=ax[2, 2].transAxes)

    fig.suptitle("SLSE vs. Steady-State SORC: NQR Sensitivity per Unit Time (NaNO2)")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
