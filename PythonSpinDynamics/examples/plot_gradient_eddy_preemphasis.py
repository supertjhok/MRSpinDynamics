"""Gradient eddy currents from coil geometry, and GRAPE pre-emphasis.

Closes the loop from coil geometry to the M5 gradient-driver model. A Maxwell
pair drives a linear gradient; a coaxial conducting shield (a cryostat-bore model)
supports eddy currents whose L/R eigenmodes (fields.eddy_modes) give the field
decay ``sum_k alpha_k exp(-t/tau_k)``. Those geometry-derived terms build an M5
``GradientDriverResponse`` -- the same object the GRAPE PGSE optimizer consumes
-- so the pre-emphasis that cancels the droop is derived from the physical coil +
shield, not hand-typed.

Part 1: eddy spectrum and the delivered-gradient step-response droop.
Part 2: pre-emphasis -- optimize the commanded gradient (bounded) so the
delivered gradient is a flat plateau, through the geometry-derived driver.
Part 3: feed the same driver into the M5 PGSE objective and show GRAPE correcting
the eddy-drooped q-vector.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.eddy_modes import EddyModes


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coil-radius-mm", type=float, default=40.0)
    parser.add_argument("--shield-radius-mm", type=float, default=60.0)
    parser.add_argument("--shield-length-mm", type=float, default=400.0)
    parser.add_argument("--shield-rings", type=int, default=16)
    parser.add_argument("--wire-radius-mm", type=float, default=2.0)
    parser.add_argument("--resistivity", type=float, default=1.7e-8, help="Shield resistivity (ohm.m); Cu ~1.7e-8.")
    parser.add_argument("--tau-rl-us", type=float, default=100.0, help="Amplifier L/R slew (us).")
    parser.add_argument("--max-terms", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    drive = coils.maxwell_pair(radius=args.coil_radius_mm * 1e-3, axis="z", n_segments=96)
    radii, centers = coils.cylindrical_shield(
        radius=args.shield_radius_mm * 1e-3, length=args.shield_length_mm * 1e-3,
        n_rings=int(args.shield_rings), axis="z",
    )
    em = EddyModes(
        radii, centers, wire_radius=args.wire_radius_mm * 1e-3,
        resistivity=args.resistivity, axis="z",
    )

    # Part 1: eddy spectrum + geometry-derived M5 gradient driver.
    spec = em.spectrum(drive, sample_point=(0, 0, 0))
    terms = em.eddy_terms(drive, max_terms=int(args.max_terms))
    driver = em.to_gradient_driver(
        drive, tau_rl=args.tau_rl_us * 1e-6, max_terms=int(args.max_terms), oversample=1,
    )
    print("Gradient eddy currents from coil + shield geometry")
    print(f"  Maxwell pair r={args.coil_radius_mm:.0f} mm, drive gradient "
          f"{spec.drive_gradient * 1e3:.3f} mT/m/A")
    print(f"  shield: {args.shield_rings} Cu rings, r={args.shield_radius_mm:.0f} mm")
    print("  geometry-derived eddy terms (alpha, tau):")
    for a, t in terms:
        print(f"    alpha = {a:.4f},  tau = {t * 1e3:.3f} ms")
    print(f"  total droop sum(alpha) = {sum(a for a, _ in terms):.4f}")

    # Part 2: pre-emphasis -- invert the driver to deliver a flat plateau.
    dt = 5e-6
    n = 400
    on = slice(20, 220)  # gradient-on window
    target = np.zeros(n)
    target[on] = 1.0
    uncompensated = driver.apply(target, dt, xp=np)

    from scipy.optimize import least_squares

    def resid(cmd_on):
        cmd = np.zeros(n)
        cmd[on] = cmd_on
        delivered = np.asarray(driver.apply(cmd, dt, xp=np))
        return delivered[on] - target[on]

    sol = least_squares(resid, np.ones(on.stop - on.start), bounds=(0.0, 4.0), max_nfev=60)
    cmd = np.zeros(n)
    cmd[on] = sol.x
    delivered = np.asarray(driver.apply(cmd, dt, xp=np))
    settled = slice(on.start + 40, on.stop)  # skip the L/R rising edge -> isolate eddy droop
    flatness_before = float(np.std(uncompensated[settled]))
    flatness_after = float(np.std(delivered[settled]))
    print("  plateau flatness (std over the settled window):")
    print(f"    uncompensated {flatness_before:.4f}  ->  pre-emphasized {flatness_after:.4f}")

    # Part 3: the same geometry-derived driver in the M5 PGSE objective.
    _pgse_qvector_demo(driver, dt)

    plt = load_matplotlib(required=True, headless=args.save is not None)
    t = np.arange(n) * dt * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].stem([t2 * 1e3 for _, t2 in terms], [a for a, _ in terms], basefmt=" ")
    axes[0].set_title("Geometry-derived eddy modes")
    axes[0].set_xlabel("tau (ms)")
    axes[0].set_ylabel("alpha (fractional droop)")
    axes[1].plot(t, target, "k--", lw=1, label="target")
    axes[1].plot(t, uncompensated, lw=1.4, label="delivered (no pre-emphasis)")
    axes[1].set_title("Eddy droop of a gradient plateau")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("gradient (norm.)")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, target, "k--", lw=1, label="target")
    axes[2].plot(t, cmd, lw=1.2, label="commanded (pre-emphasized)")
    axes[2].plot(t, delivered, lw=1.4, label="delivered")
    axes[2].set_title("Pre-emphasis through the M5 driver")
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("gradient (norm.)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")
    else:
        plt.show()


def _pgse_qvector_demo(driver, dt) -> None:
    """Feed the geometry-derived driver into the M5 PGSE objective (q-vector)."""

    try:
        from spin_dynamics.coupling.systems import coupled_spin_system
        from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
        from spin_dynamics.optimal_control.diffusion import make_pgse_objective
        from spin_dynamics.optimal_control.hamiltonians import (
            coupled_spin_control_model,
            gradient_control_operator,
        )
        from spin_dynamics.optimization._bounded import scipy_maximize_with_grad
    except Exception:  # pragma: no cover
        return
    if not JAX_AVAILABLE:
        print("  (M5 PGSE tie-in skipped: jax not installed)")
        return

    psi_up = np.array([1.0, 0.0], dtype=np.complex128)
    i_plus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    g_scale = 2000.0
    base = coupled_spin_control_model(coupled_spin_system([0.0], [[0.0]]))
    hg0 = gradient_control_operator(coupled_spin_system([0.0], [[0.0]])) * g_scale
    drift_b, hx_b, hy_b, hg_b, weights, offs = [], [], [], [], [], []
    for off in (-150.0, 0.0, 150.0):
        d = coupled_spin_control_model(coupled_spin_system([off], [[0.0]])).h_drift
        for r in (-1.0, 0.0, 1.0):
            drift_b.append(d)
            hx_b.append(base.h_x)
            hy_b.append(base.h_y)
            hg_b.append(r * hg0)
            weights.append(1.0)
            offs.append(off)
    n = 16
    amp = np.zeros(n)
    amp[:3] = 4000.0
    amp[7:10] = 4000.0
    gmask = np.zeros(n)
    gmask[3:7] = 1.0
    gmask[10:14] = 1.0
    sign = np.ones(n)
    sign[10:] = -1.0
    q_target = 4 * (dt if np.isscalar(dt) else 5e-6)  # unit gradient over the encode lobe
    vg = make_pgse_objective(
        drift_batch=drift_b, hx_batch=hx_b, hy_batch=hy_b, hgrad_batch=hg_b,
        psi0=psi_up, detection_operator=i_plus, weights=np.array(weights), offsets_hz=np.array(offs),
        dt=5e-6, n_segments=n, amplitude_template=amp, gradient_mask=gmask, coherence_sign=sign,
        encode_end_segment=7, q_target=q_target, acquisition_points=33, acquisition_dt=2e-6,
        q_weight=1e6, refocus_weight=1e6, gradient_response=driver,
    )
    x0 = np.concatenate([np.zeros(n), np.ones(n)])
    v0 = vg(x0)[0]
    bounds = [(-4 * np.pi, 4 * np.pi)] * n + [(-3.0, 3.0)] * n
    run = scipy_maximize_with_grad(vg, x0, bounds=bounds, options={"maxiter": 80})
    print("  M5 PGSE with the geometry-derived driver:")
    print(f"    objective {v0:.4f} -> {run.best_score:.4f} "
          f"(GRAPE pre-emphasizes the physically-predicted eddy droop)")


if __name__ == "__main__":
    main()
