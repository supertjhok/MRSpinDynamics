"""GRAPE PGSE: correct q-vector + max detected SNR on realistic hardware.

The whole-chain diffusion capstone (design note
``docs/optimal_control_pgse_diffusion.md``). A pulsed-gradient spin echo
(90 - encode lobe - 180 - refocus lobe - echo) is optimized over a GROSSLY
inhomogeneous ``B0`` field (offset spread several times the nutation rate) plus a
``B1`` spread and a position grid, with a tuned RF probe on BOTH transmit and
receive and a gradient driver that has ``L/R`` slew plus eddy-current droop.
GRAPE co-optimizes the RF phase (90 + 180) and the gradient waveform against:

    maximize  detected_SNR  -  q_weight*(q_encode - q*)^2  -  refocus_weight*q_echo^2

Two effects, both real and large here:

* **Broadband phase-only refocusing.** A hard 180 refocuses only over ~+-nutation,
  so across a wide band the echo collapses. GRAPE finds a phase-modulated
  refocusing pulse (the RP2 / symmetric-phase-alternating family) that recovers
  the echo far beyond the hard-pulse bandwidth -- a several-fold detected-SNR
  gain. Multistart is essential: a rectangular (phase=0) seed against a symmetric
  offset ensemble sits at an exact saddle, so a single start never moves the
  phase.
* **Eddy-compensated q-vector.** The gradient driver droops the delivered
  gradient, so the naive encode moment misses ``q*`` and fails to refocus; GRAPE
  pre-emphasizes the commanded gradient to hit ``q*`` and null the residual.

Requires the optional ``jax`` extra. Prints a summary always; ``--save`` writes a
six-panel figure: the optimized phase pulse, gradient pre-emphasis, delivered
q(t), naive vs optimized echo heatmaps over a fine held-out ``(B0, B1)`` grid,
and the refocusing-bandwidth profile.
"""

from __future__ import annotations

import argparse
from math import ceil

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling.systems import coupled_spin_system
from spin_dynamics.optimal_control._jax_propagation import (
    JAX_AVAILABLE,
    propagate_batched_grad_controls_numpy,
)
from spin_dynamics.optimal_control.control_response import (
    GradientDriverResponse,
    ReceiverResponse,
    TailSpec,
    TunedProbeResponse,
)
from spin_dynamics.optimal_control.hamiltonians import (
    coupled_spin_control_model,
    gradient_control_operator,
)

_PSI_UP = np.array([1.0, 0.0], dtype=np.complex128)
_I_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


def _pgse_layout(n):
    """Lay out a PGSE: 90 - gap - encode - gap - 180(2x) - gap - refocus - gap(echo).

    Returns ``(amp_template_unit, gradient_mask, coherence_sign, encode_end,
    p90, lobe)``. The 180 window is twice the 90 (same amplitude -> twice the
    flip); ``p90`` sets the nutation rate so the 90 window is a true 90 degrees.
    """

    amp = np.zeros(n)
    mask = np.zeros(n)
    sign = np.ones(n)
    p90 = max(4, n // 9)          # longer pulse -> lower nutation -> finite bandwidth
    lobe = max(4, n // 10)
    gap = max(3, n // 12)          # real dead time between pulses and lobes
    idx = gap
    amp[idx : idx + p90] = 1.0  # 90
    idx += p90 + gap
    enc0 = idx
    mask[enc0 : enc0 + lobe] = 1.0  # encode lobe
    idx += lobe
    encode_end = idx
    idx += gap
    b180 = idx
    amp[b180 : b180 + 2 * p90] = 1.0  # 180 (twice the 90)
    flip = b180 + p90  # sign flips at the 180 center
    idx += 2 * p90 + gap
    mask[idx : idx + lobe] = 1.0  # refocus lobe
    sign[flip:] = -1.0
    return amp, mask, sign, encode_end, p90, lobe


def _build_ensemble(positions, offsets, b1s, g_scale):
    base = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    hx0, hy0 = base.h_x, base.h_y
    hgrad0 = gradient_control_operator(coupled_spin_system([0.0], [[0.0]])) * g_scale
    drift_b, hx_b, hy_b, hgrad_b, weights, offs = [], [], [], [], [], []
    for off in offsets:
        drift = coupled_spin_control_model(coupled_spin_system([off], [[0.0]])).h_drift
        for b1 in b1s:
            for r in positions:
                drift_b.append(drift)
                hx_b.append(b1 * hx0)
                hy_b.append(b1 * hy0)
                hgrad_b.append(r * hgrad0)
                weights.append(1.0)
                offs.append(off)
    return (drift_b, hx_b, hy_b, hgrad_b, np.array(weights), np.array(offs))


def _deliver_numpy(amp_template, phase, gradient_cmd, dt, rf, driver):
    """NumPy reconstruction of the delivered (fine-grid) controls, for reporting/plots."""

    s = max(rf.oversample, driver.oversample)
    dt_fine = dt / s
    tail_s = max(rf.tail.seconds(rf.dominant_tau), driver.tail.seconds(driver.dominant_tau))
    n_tail = ceil(tail_s / dt_fine)
    n = amp_template.size
    n_fine = n * s + n_tail

    u = amp_template * np.exp(1j * phase)
    u_fine = np.concatenate([np.repeat(u, s), np.zeros(n_tail, complex)])
    freqs = np.fft.fftfreq(n_fine, d=dt_fine)
    delivered = np.fft.ifft(np.fft.fft(u_fine) * rf.transfer(freqs, xp=np))
    amp_f = np.abs(delivered)
    phase_f = np.angle(delivered)

    g_fine = np.concatenate([np.repeat(gradient_cmd, s), np.zeros(n_tail)])
    grad_f = driver.apply(g_fine, dt_fine, xp=np)
    return amp_f, phase_f, grad_f, dt_fine, s, n_tail


def _evaluate(amp_template, phase, gradient_cmd, dt, ens, rf, driver, rx, sign,
              encode_end, acq_pts, acq_dt, noise):
    """Coherences, per-(B0,B1) echo amplitude, detected SNR, and q-moments (NumPy)."""

    drift_b, hx_b, hy_b, hgrad_b, weights, offs = ens
    amp_f, phase_f, grad_f, dt_fine, s, n_tail = _deliver_numpy(
        amp_template, phase, gradient_cmd, dt, rf, driver
    )
    psi0_batch = np.broadcast_to(_PSI_UP, (len(drift_b), 2))
    psis = propagate_batched_grad_controls_numpy(
        np.asarray(drift_b), np.asarray(hx_b), np.asarray(hy_b), np.asarray(hgrad_b),
        amp_f, phase_f, grad_f, dt_fine, psi0_batch,
    )
    coh = np.array([np.vdot(psis[k], _I_PLUS @ psis[k]) for k in range(len(psis))])

    # Acquired echo (coherent sum over ALL members), Rx-filtered, matched-filter energy.
    tau = (np.arange(acq_pts) - (acq_pts - 1) / 2.0) * acq_dt
    signal = np.sum((weights * coh)[:, None] * np.exp(2j * np.pi * offs[:, None] * tau[None, :]), axis=0)
    rxf = np.fft.ifft(np.fft.fft(signal) * rx.transfer(np.fft.fftfreq(acq_pts, d=acq_dt), xp=np))
    snr = np.sqrt(np.sum(np.abs(rxf) ** 2) * acq_dt) / noise

    # q-vector of the delivered gradient (effective sign on the fine grid).
    sign_fine = np.concatenate([np.repeat(sign, s), np.full(n_tail, sign[-1])])
    q_of_t = np.cumsum(grad_f * sign_fine) * dt_fine
    q_encode = q_of_t[encode_end * s - 1]
    q_residual = q_of_t[-1]
    return coh, snr, q_of_t, q_encode, q_residual, (amp_f, grad_f, dt_fine)


def _echo_amplitude_grid(amp_template_hz, phase, grad_cmd, dt, n, positions,
                         eval_offsets, eval_b1s, g_scale, rf, driver):
    """Echo amplitude |sum_pos <I+>| over a fine held-out ``(B0, B1)`` grid (JAX).

    A single fast forward evaluation -- separate from (and finer than) the coarse
    training ensemble -- so the inhomogeneity structure of the heatmap is well
    resolved without slowing the optimization.
    """

    import jax.numpy as jnp

    from spin_dynamics.optimal_control._jax_propagation import propagate_batched_grad_controls
    from spin_dynamics.optimal_control.control_response import build_control_delivery

    drift_b, hx_b, hy_b, hgrad_b, _w, _o = _build_ensemble(positions, eval_offsets, eval_b1s, g_scale)
    deliver, _nf, _dtf = build_control_delivery(n, dt, rf_response=rf, gradient_response=driver)
    a, pf, gf, dte = deliver(jnp.asarray(amp_template_hz), jnp.asarray(phase), jnp.asarray(grad_cmd))
    psi0b = jnp.broadcast_to(jnp.asarray(_PSI_UP), (len(drift_b), 2))
    psis = propagate_batched_grad_controls(
        jnp.asarray(drift_b), jnp.asarray(hx_b), jnp.asarray(hy_b), jnp.asarray(hgrad_b),
        a, pf, gf, dte, psi0b, method="expm",
    )
    coh = np.asarray(jnp.einsum("ki,ij,kj->k", jnp.conj(psis), jnp.asarray(_I_PLUS), psis))
    coh = coh.reshape(len(eval_offsets), len(eval_b1s), len(positions))
    return np.abs(coh.sum(axis=2))  # coherent sum over the position grid


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=63)
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--starts", type=int, default=6, help="Random phase restarts (escape the saddle).")
    parser.add_argument("--offsets", type=int, default=5, help="B0 points in the (coarse) training ensemble.")
    parser.add_argument("--b1s", type=int, default=3, help="B1 points in the (coarse) training ensemble.")
    parser.add_argument("--eval-offsets", type=int, default=31, help="B0 points in the fine heatmap grid.")
    parser.add_argument("--eval-b1s", type=int, default=23, help="B1 points in the fine heatmap grid.")
    parser.add_argument("--offset-hz", type=float, default=8000.0,
                        help="Half-width of the B0 spread (grossly inhomogeneous: several x nutation).")
    parser.add_argument("--b1-min", type=float, default=0.7)
    parser.add_argument("--b1-max", type=float, default=1.3)
    parser.add_argument("--probe-q", type=float, default=110.0)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print("GRAPE requires the optional 'jax' extra (`pip install -e .[jax]`).")
        return

    from spin_dynamics.optimal_control.diffusion import make_pgse_objective
    from spin_dynamics.optimization._bounded import scipy_maximize_with_grad

    n = int(args.segments)
    dt = 1e-5
    g_scale = 2000.0
    amp_template, grad_mask, sign, encode_end, p90, lobe = _pgse_layout(n)
    positions = np.linspace(-1.0, 1.0, int(args.positions))
    offsets = np.linspace(-args.offset_hz, args.offset_hz, int(args.offsets))
    b1s = np.linspace(args.b1_min, args.b1_max, int(args.b1s))
    ens = _build_ensemble(positions, offsets, b1s, g_scale)
    _drift, _hx, _hy, _hg, weights, offs = ens

    rf = TunedProbeResponse.from_quality_factor(f0_hz=1e6, quality_factor=args.probe_q, oversample=4)
    rx = ReceiverResponse(tau=rf.tau)  # same probe on receive (reciprocity)
    driver = GradientDriverResponse(tau_rl=3e-6, eddy_terms=((0.25, 5e-5),), oversample=4,
                                    tail=TailSpec(factor=4.0))
    # Nutation rate so the p90-segment window is exactly a 90 deg pulse.
    nutation_hz = 0.25 / (p90 * dt)
    amp_template_hz = amp_template * nutation_hz

    acq_pts, acq_dt, noise = 41, 2e-6, 1.0
    # q target: what a unit gradient control encodes over the lobe on an ideal
    # driver -- the eddy droop means the optimizer must pre-emphasize to reach it.
    q_target = 1.0 * lobe * dt

    vg = make_pgse_objective(
        drift_batch=_drift, hx_batch=_hx, hy_batch=_hy, hgrad_batch=_hg,
        psi0=_PSI_UP, detection_operator=_I_PLUS, weights=weights, offsets_hz=offs,
        dt=dt, n_segments=n, amplitude_template=amp_template_hz, gradient_mask=grad_mask,
        coherence_sign=sign, encode_end_segment=encode_end, q_target=q_target,
        acquisition_points=acq_pts, acquisition_dt=acq_dt, noise=noise,
        q_weight=2e9, refocus_weight=2e9,
        rf_response=rf, gradient_response=driver, rx_response=rx,
    )

    # Naive baseline: rectangular RF phase (hard 90 + hard 180), unit gradient.
    phase0 = np.zeros(n)
    grad0 = grad_mask.copy()
    x0 = np.concatenate([phase0, grad0])
    v0 = vg(x0)[0]

    # Multistart is not optional: a rectangular (phase=0) seed against a
    # symmetric offset ensemble sits at an EXACT saddle -- mirrored offsets'
    # phase gradients cancel, so a single start never moves the phase (it would
    # only fix the gradient channel). Random phase starts escape it and let GRAPE
    # find a broadband phase-modulated refocusing pulse (RP2 / symmetric-phase-
    # alternating family), which is what recovers the echo across the wide band.
    bounds = [(-4 * np.pi, 4 * np.pi)] * n + [(-3.0, 3.0)] * n
    rng = np.random.default_rng(int(args.seed))
    best = None
    for trial in range(int(args.starts)):
        if trial == 0:
            seed = x0
        else:
            seed = np.concatenate([rng.uniform(-np.pi, np.pi, n), grad0])
        run = scipy_maximize_with_grad(vg, seed, bounds=bounds, options={"maxiter": int(args.maxiter)})
        if best is None or run.best_score > best.best_score:
            best = run
    xopt = best.best_x
    phase_opt = xopt[:n]
    grad_opt = xopt[n:] * grad_mask

    ev = dict(dt=dt, ens=ens, rf=rf, driver=driver, rx=rx, sign=sign,
              encode_end=encode_end, acq_pts=acq_pts, acq_dt=acq_dt, noise=noise)
    _c0, snr0, q0, qe0, qr0, _w0 = _evaluate(amp_template_hz, phase0, grad0, **ev)
    _c1, snr1, q1, qe1, qr1, wav1 = _evaluate(amp_template_hz, phase_opt, grad_opt, **ev)

    print("GRAPE PGSE: correct q-vector + detected SNR (tuned probe Tx+Rx, eddy driver)")
    print(f"  ensemble: {len(offs)} members ({args.positions} pos x {args.offsets} B0 x {args.b1s} B1)"
          f",  B0 +/-{args.offset_hz:.0f} Hz,  probe Q={args.probe_q:.0f} (ring-down {rf.tau*1e6:.1f} us)")
    print(f"  q target: {q_target:.3e}")
    print(f"  {'':16s}{'naive':>14s}{'optimized':>14s}")
    print(f"  {'detected SNR':16s}{snr0:>14.4f}{snr1:>14.4f}")
    print(f"  {'q_encode':16s}{qe0:>14.3e}{qe1:>14.3e}")
    print(f"  {'q_encode err':16s}{qe0 - q_target:>14.2e}{qe1 - q_target:>14.2e}")
    print(f"  {'q_residual':16s}{qr0:>14.2e}{qr1:>14.2e}")
    print(f"  nutation {nutation_hz:.0f} Hz vs B0 +/-{args.offset_hz:.0f} Hz "
          f"(offset/nutation = {args.offset_hz / nutation_hz:.1f}x)")
    print(f"  objective: {v0:.4f} -> {best.best_score:.4f}  [{int(args.starts)} starts]")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        amp_f1, grad_f1, dt_fine = wav1
        s = max(rf.oversample, driver.oversample)
        t_fine = np.arange(grad_f1.size) * dt_fine * 1e6

        # (0,0) what GRAPE designed: the optimized phase-only refocusing pulse.
        # Amplitude is the fixed 90/180 template (shaded); the OPTIMIZED PHASE
        # across the pulse windows is the broadband refocusing pattern.
        seg_t = (np.arange(n) + 0.5) * dt * 1e6
        in_pulse = amp_template > 0
        # NaN outside the pulse windows so the line breaks across the dead gaps
        # (phase is meaningless where there is no RF).
        phase_disp = np.where(in_pulse, np.mod(phase_opt, 2 * np.pi) / np.pi, np.nan)
        axes[0, 0].fill_between(np.arange(n) * dt * 1e6, 0, amp_template * 1.0,
                                step="post", color="0.85", label="pulse windows (90 / 180)")
        axes[0, 0].step(seg_t, phase_disp, where="mid", color="C1", lw=1.4, label="optimized phase")
        axes[0, 0].set_title("GRAPE phase-only refocusing pulse")
        axes[0, 0].set_xlabel("time (us)")
        axes[0, 0].set_ylabel("phase (pi rad)  /  windows")
        axes[0, 0].legend(fontsize=8)

        # (0,1) gradient: commanded (pre-emphasized) vs delivered (eddy)
        cmd_fine = np.concatenate([np.repeat(grad_opt, s), np.zeros(grad_f1.size - n * s)])
        axes[0, 1].plot(t_fine, cmd_fine, label="commanded", lw=1.2)
        axes[0, 1].plot(t_fine, grad_f1, label="delivered (eddy)", lw=1.2)
        axes[0, 1].set_title("Gradient: pre-emphasis vs eddy droop")
        axes[0, 1].set_xlabel("time (us)")
        axes[0, 1].set_ylabel("gradient (control units)")
        axes[0, 1].legend(fontsize=8)

        # (0,2) delivered q(t): encode + refocus
        axes[0, 2].axhline(q_target, ls="--", color="k", lw=1, label="q*")
        axes[0, 2].plot(np.arange(q0.size) * dt_fine * 1e6, q0, label="naive", lw=1.2)
        axes[0, 2].plot(np.arange(q1.size) * dt_fine * 1e6, q1, label="optimized", lw=1.2)
        axes[0, 2].set_title("Delivered q(t): encode + refocus")
        axes[0, 2].set_xlabel("time (us)")
        axes[0, 2].set_ylabel("q (effective)")
        axes[0, 2].legend(fontsize=8)

        # (1,0)-(1,1) fine held-out echo-amplitude heatmaps over (B0, B1)
        eval_off = np.linspace(-args.offset_hz, args.offset_hz, int(args.eval_offsets))
        eval_b1 = np.linspace(args.b1_min, args.b1_max, int(args.eval_b1s))
        grid_kw = dict(dt=dt, n=n, positions=positions, eval_offsets=eval_off, eval_b1s=eval_b1,
                       g_scale=g_scale, rf=rf, driver=driver)
        naive_grid = _echo_amplitude_grid(amp_template_hz, phase0, grad0, **grid_kw)
        opt_grid = _echo_amplitude_grid(amp_template_hz, phase_opt, grad_opt, **grid_kw)
        vmax = max(naive_grid.max(), opt_grid.max())
        extent = [eval_off[0], eval_off[-1], eval_b1[0], eval_b1[-1]]
        for ax, grid, title in ((axes[1, 0], naive_grid, "naive"), (axes[1, 1], opt_grid, "optimized")):
            im = ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=vmax)
            ax.set_title(f"Echo amplitude over (B0, B1): {title}")
            ax.set_xlabel("B0 offset (Hz)")
            ax.set_ylabel("B1 scale")
            fig.colorbar(im, ax=ax)

        # (1,2) refocusing bandwidth: echo amplitude vs B0 at B1 = 1 (the money panel)
        b1_idx = int(np.argmin(np.abs(eval_b1 - 1.0)))
        axes[1, 2].plot(eval_off, naive_grid[:, b1_idx], label="naive (hard 180)", lw=1.4, color="gray")
        axes[1, 2].plot(eval_off, opt_grid[:, b1_idx], label="GRAPE refocusing", lw=1.4, color="C0")
        axes[1, 2].axvspan(-args.offset_hz, args.offset_hz, color="C0", alpha=0.06)
        axes[1, 2].set_title(f"Refocusing bandwidth at B1=1\n(SNR {snr0:.3f} -> {snr1:.3f})")
        axes[1, 2].set_xlabel("B0 offset (Hz)")
        axes[1, 2].set_ylabel("echo amplitude")
        axes[1, 2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
