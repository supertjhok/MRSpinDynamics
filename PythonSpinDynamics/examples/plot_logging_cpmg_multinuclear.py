"""Jasper-Jackson NMR logging: CPMG sensitive regions, signal, SNR, and 23Na.

Builds on ``plot_logging_coil_loading.py``. The tool is a Jasper-Jackson magnet:
two opposing cylindrical (SmCo) permanent magnets facing N-N, whose fields cancel
on the midplane and leave a low-gradient "sweet spot" at a depth set by the
face-to-face separation. The RF frequency is chosen to resonate at the sweet-spot
field, so the 1H sensitive region is a thick, low-gradient shell in the formation.
The solenoid coil (in the gap) sets B1.

For every voxel the asymptotic CPMG echo -- the fraction of equilibrium
magnetization surviving the refocusing train -- is computed from the local
off-resonance and nutation, weighting the sensitive region. The echo is summed
over the region (reciprocity), band-limited by the tuned probe, and the per-echo
SNR is estimated against the coil+brine Johnson noise (``coil_loading``; receiver
assumed noiseless).

Multinuclear 23Na: saturated brine contains 23Na, which (gamma_Na/gamma_H ~ 0.26)
resonates at the same RF frequency where B0 is ~3.8x higher -- an inner shell,
closer to the magnets. **23Na is spin-3/2 and is treated with the proper (2I+1)
machinery**; it also gets non-optimal flip angles, because the pulse durations are
calibrated for 1H and 23Na simply nutates at its own (lower-gamma) rate. Its
sensitive region and fractional contribution to the total signal are plotted.

Physics notes: relaxation and diffusion are ignored (asymptotic CPMG echo). In
motionally-narrowed brine the 23Na quadrupolar coupling averages to zero, so the
Hamiltonian is linear (Zeeman+RF); a consequence is that the *detectable* echo
profile is spin-independent (verified: spin-1/2 and spin-3/2 agree), so 23Na
differs only through its gamma (shell + flip angles) and the I(I+1)/concentration
signal scaling -- not the echo shape.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.magnetostatics import FiniteMagnetRod, biot_savart, finite_magnet_array_b0
from spin_dynamics.fields.quasistatic import coil_inductance, coil_loading

HBAR = 1.054571817e-34
KB = 1.380649e-23
GAMMA_H = 2.675222e8
GAMMA_NA = 7.080848e7   # 23Na (spin 3/2)


def _spin_ops(spin_i):
    d = int(round(2 * spin_i + 1))
    m = np.arange(spin_i, -spin_i - 1, -1.0)
    iz = np.diag(m).astype(complex)
    ip = np.zeros((d, d), dtype=complex)
    for k in range(d - 1):
        mm = m[k + 1]
        ip[k, k + 1] = np.sqrt(spin_i * (spin_i + 1) - mm * (mm + 1))
    ix = 0.5 * (ip + ip.conj().T)
    iy = (ip - ip.conj().T) / 2j
    return ix, iy, iz


def cpmg_echo(spin_i, offset, omega1, t90, t180, tau_half):
    """Asymptotic CPMG transverse echo fraction for a spin-I isochromat.

    General (2I+1)-level treatment: 90_x excitation, then a tau/2 - 180_y - tau/2
    refocusing cycle. After many cycles only the part of the excited density that
    is secular under the cycle propagator survives (Schur basis handles the
    on-resonance eigenvalue degeneracies); the echo is its transverse expectation
    ``|<I+>|`` normalized to the initial ``<Iz>``. Relaxation ignored.
    """

    from scipy.linalg import expm, schur

    ix, iy, iz = _spin_ops(spin_i)
    u90 = expm(-1j * (omega1 * ix + offset * iz) * t90)
    rho = u90 @ iz @ u90.conj().T
    u180 = expm(-1j * (omega1 * iy + offset * iz) * t180)
    ufree = expm(-1j * offset * iz * tau_half)
    cycle = ufree @ u180 @ ufree
    tmat, v = schur(cycle, output="complex")   # v unitary even for degeneracies
    phase = np.angle(np.diag(tmat))
    rp = v.conj().T @ rho @ v
    dphi = np.abs(phase[:, None] - phase[None, :])
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    rp = np.where(dphi < 1e-6, rp, 0.0)          # secular: keep degenerate blocks
    rho_surv = v @ rp @ v.conj().T
    mx = np.trace(ix @ rho_surv).real
    my = np.trace(iy @ rho_surv).real
    return float(np.hypot(mx, my) / np.trace(iz @ iz).real)


def _echo_table(spin_i, t90, t180, tau_half, offset_axis, omega1_axis):
    """Precompute echo(offset, omega1) and return a bilinear interpolator."""

    from scipy.interpolate import RegularGridInterpolator

    table = np.empty((offset_axis.size, omega1_axis.size))
    for i, off in enumerate(offset_axis):
        for j, w1 in enumerate(omega1_axis):
            table[i, j] = cpmg_echo(spin_i, off, w1, t90, t180, tau_half)
    return RegularGridInterpolator((offset_axis, omega1_axis), table,
                                   bounds_error=False, fill_value=0.0)


def _jj_magnet(remanence, length, radius, gap):
    zc = gap / 2.0 + length / 2.0
    return [
        FiniteMagnetRod(center=(0, 0, zc), length=length, br=(0, 0, -remanence), shape="cylinder", radius=radius),
        FiniteMagnetRod(center=(0, 0, -zc), length=length, br=(0, 0, remanence), shape="cylinder", radius=radius),
    ]


def _curie_m0(n_density, gamma, spin_i, b0, temperature):
    return n_density * gamma**2 * HBAR**2 * spin_i * (spin_i + 1.0) * b0 / (3.0 * KB * temperature)


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remanence", type=float, default=1.05, help="SmCo remanence (T).")
    parser.add_argument("--magnet-length-cm", type=float, default=30.0)
    parser.add_argument("--magnet-radius-cm", type=float, default=8.0)
    parser.add_argument("--gap-cm", type=float, default=34.0, help="Face-to-face separation (tunes depth).")
    parser.add_argument("--pulse90-us", type=float, default=25.0,
                        help="1H 90-degree pulse length (hardware spec); the peak current follows.")
    parser.add_argument("--echo-spacing-us", type=float, default=600.0)
    parser.add_argument("--temperature-k", type=float, default=350.0)
    parser.add_argument("--brine-conductivity", type=float, default=10.0)
    parser.add_argument("--borehole-radius-cm", type=float, default=11.0)
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--nz", type=int, default=140)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    length = args.magnet_length_cm * 1e-2
    radius = args.magnet_radius_cm * 1e-2
    gap = args.gap_cm * 1e-2
    rods = _jj_magnet(args.remanence, length, radius, gap)

    # Field map in the r-z plane (axisymmetric); avoid the magnet interior (r<radius, |z|>gap/2).
    xs = np.linspace(0.02, 0.34, int(args.nx))
    zs = np.linspace(-0.20, 0.20, int(args.nz))
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1)
    b0_vec = finite_magnet_array_b0(pts, rods)
    b0_mag = np.linalg.norm(b0_vec, axis=-1)
    inside_magnet = (X < radius) & (np.abs(Z) > gap / 2.0)
    b0_mag = np.where(inside_magnet, np.nan, b0_mag)

    # Sweet spot: the midplane |B0| maximum -> set the RF frequency to resonate there.
    mid = int(np.argmin(np.abs(zs)))
    prof = np.nan_to_num(b0_mag[:, mid])
    i_sweet = int(np.argmax(prof))
    r_sweet = xs[i_sweet]
    b_sweet = prof[i_sweet]
    f_rf = GAMMA_H * b_sweet / (2.0 * np.pi)
    w_rf = 2.0 * np.pi * f_rf

    # Coil (solenoid) in the gap, radius near the borehole wall for B1 efficiency
    # at the (outside-the-borehole) sweet spot; B1 and its transverse component.
    coil_radius = min(0.09, args.borehole_radius_cm * 1e-2 * 0.82)
    coil_turns = 30
    coil = coils.solenoid(radius=coil_radius, length=min(0.30, gap * 0.85), turns=coil_turns, axis="z", n_segments=48)
    b1_vec = biot_savart(pts, coil, 1.0)
    b0_hat = np.where(np.isfinite(b0_mag)[..., None], b0_vec / np.where(b0_mag[..., None] > 0, b0_mag[..., None], 1), 0.0)
    b1_par = np.sum(b1_vec * b0_hat, axis=-1)[..., None] * b0_hat
    b1_perp = np.linalg.norm(b1_vec - b1_par, axis=-1)   # T/A

    # Fixed 1H pulse lengths (hardware spec); the peak coil current follows from
    # the B1 efficiency at the sweet spot.
    t90 = args.pulse90_us * 1e-6
    t180 = 2.0 * t90
    w1_cal = (np.pi / 2.0) / t90
    b1_perp_sweet = float(b1_perp[i_sweet, mid])
    coil_current = w1_cal / (GAMMA_H * b1_perp_sweet)
    tau_half = max(0.5 * args.echo_spacing_us * 1e-6 - 0.5 * t180, 0.0)

    # Echo lookup tables (fixed pulses). 1H spin-1/2, 23Na spin-3/2.
    off_axis = np.linspace(-10 * w1_cal, 10 * w1_cal, 121)
    w1_axis = np.linspace(0.0, 3.0 * w1_cal, 61)
    table_h = _echo_table(0.5, t90, t180, tau_half, off_axis, w1_axis)
    table_na = _echo_table(1.5, t90, t180, tau_half, off_axis, w1_axis)

    # Brine spin densities (saturated NaCl ~ 5.3 M Na, ~98 M H).
    n_h = 98.6e3 * 6.02214e23
    n_na = 5.3e3 * 6.02214e23

    def nucleus(gamma, spin_i, n_density, table):
        b_res = w_rf / gamma
        offset = gamma * (b0_mag - b_res)
        omega1 = gamma * b1_perp * coil_current
        # Voxel-average the (thin) resonance over the radial offset span.
        d_off = np.abs(np.gradient(np.nan_to_num(offset), dx, axis=0)) * dx
        n_sub = 15
        sub = np.linspace(-0.5, 0.5, n_sub)
        off_sub = offset[..., None] + sub * np.maximum(d_off, 1e-30)[..., None]
        w1_sub = np.broadcast_to(omega1[..., None], off_sub.shape)
        echo = table(np.stack([off_sub, w1_sub], axis=-1)).mean(axis=-1)
        echo = np.nan_to_num(np.where(np.isfinite(b0_mag), echo, 0.0))
        m0 = _curie_m0(n_density, gamma, spin_i, np.nan_to_num(b0_mag), args.temperature_k)
        dv = 2.0 * np.pi * X * dx * dz
        signal_density = w_rf * b1_perp * m0 * echo       # V per m^3
        weight = signal_density * dv
        flip = omega1 * t90 / (np.pi / 2.0) * 90.0         # deg, nominal 90 at calibration
        return dict(b_res=b_res, offset=offset, omega1=omega1, echo=echo,
                    weight=np.nan_to_num(weight), total=float(np.nansum(weight)),
                    sensitivity=np.nan_to_num(signal_density), flip=flip)

    proton = nucleus(GAMMA_H, 0.5, n_h, table_h)
    sodium = nucleus(GAMMA_NA, 1.5, n_na, table_na)
    total = proton["total"] + sodium["total"]
    na_fraction = sodium["total"] / total if total > 0 else 0.0

    # Coil loading -> Johnson noise.
    r_bh = args.borehole_radius_cm * 1e-2
    radii = np.full(coil_turns, coil_radius)
    centers = np.zeros((coil_turns, 3))
    centers[:, 2] = np.linspace(-0.15, 0.15, coil_turns)
    inductance = coil_inductance(radii, centers, wire_radius=1e-3)
    ng = 23
    axg = np.linspace(-r_bh * 1.1, r_bh * 1.1, ng)
    azg = np.linspace(-0.2, 0.2, ng)
    dgx = axg[1] - axg[0]
    dgz = azg[1] - azg[0]
    GX, GY, GZ = np.meshgrid(axg, axg, azg, indexing="ij")
    rho3 = np.sqrt(GX**2 + GY**2)
    brine = (rho3 >= coil_radius) & (rho3 <= r_bh) & (np.abs(GZ) <= 0.18)
    loading = coil_loading(np.stack([GX, GY, GZ], axis=-1), coil, conductivity=args.brine_conductivity,
                           mask=brine, spacing=(dgx, dgx, dgz), frequencies=[f_rf],
                           inductance=inductance, coil_resistance=0.5 * np.sqrt(f_rf / 1e6))
    r_total = float(loading.reflected_resistance[0] + 0.5 * np.sqrt(f_rf / 1e6))
    q_loaded = float(loading.q_loaded[0])
    bandwidth = f_rf / q_loaded
    rf_power = 0.5 * coil_current**2 * r_total   # peak-pulse power into R_total (W)
    brine_power_fraction = float(loading.reflected_resistance[0]) / r_total

    # Received echo through the tuned probe (coherent sum over offsets).
    n_acq = 201
    t_acq = np.linspace(-2e-3, 2e-3, n_acq)
    def echo_wave(res):
        w = res["weight"].ravel()
        off = np.nan_to_num(res["offset"]).ravel()
        keep = w > w.max() * 1e-3 if w.max() > 0 else np.zeros_like(w, bool)
        return np.sum(w[keep, None] * np.cos(off[keep, None] * t_acq[None, :]), axis=0)
    echo_h = echo_wave(proton)
    echo_na = echo_wave(sodium)
    signal_peak = float(np.max(np.abs(echo_h + echo_na)))
    noise_rms = np.sqrt(4.0 * KB * args.temperature_k * r_total * bandwidth)
    snr = signal_peak / noise_rms if noise_rms > 0 else 0.0

    na_flip_med = float(np.nanmedian(sodium["flip"][sodium["echo"] > 0.05 * sodium["echo"].max()])) if np.any(sodium["echo"] > 0) else 0.0
    print("Jasper-Jackson NMR logging: CPMG sensitive regions, signal, SNR, 23Na")
    print(f"  magnet: 2x SmCo (Br={args.remanence} T, L={args.magnet_length_cm:.0f} cm) gap {args.gap_cm:.0f} cm")
    print(f"  sweet spot: r={r_sweet * 100:.1f} cm, |B0|={b_sweet:.4f} T -> f_RF={f_rf / 1e6:.3f} MHz")
    print(f"  1H pulses: 90 = {t90 * 1e6:.1f} us, 180 = {t180 * 1e6:.1f} us "
          f"(nutation {w1_cal / 2 / np.pi / 1e3:.1f} kHz at the sweet spot)")
    print(f"  coil: {coil_turns}-turn solenoid r={coil_radius * 100:.0f} cm, "
          f"B1 efficiency {b1_perp_sweet * 1e6:.1f} uT/A at the sweet spot")
    print(f"  peak coil current: {coil_current:.0f} A;  RF power: {rf_power / 1e3:.2f} kW peak-pulse "
          f"({brine_power_fraction * 100:.0f}% dissipated in the brine)")
    print(f"  23Na shell |B0|={sodium['b_res']:.4f} T (inner); 23Na flip ~{na_flip_med:.0f} deg (proton-calibrated pulse)")
    print(f"  received echo EMF: 1H {proton['total'] * 1e9:.2f} nV, 23Na {sodium['total'] * 1e9:.2f} nV")
    print(f"  23Na fractional contribution: {na_fraction * 100:.1f} %")
    print(f"  coil loading: R_total {r_total:.2f} ohm, Q_loaded {q_loaded:.0f}, bandwidth {bandwidth / 1e3:.1f} kHz")
    print(f"  noise (Johnson, T={args.temperature_k:.0f} K, noiseless Rx): {noise_rms * 1e9:.3f} nV")
    print(f"  estimated per-echo SNR: {snr:.1f}")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ext = [xs[0] * 100, xs[-1] * 100, zs[0] * 100, zs[-1] * 100]
        b0_plot = np.nan_to_num(b0_mag, nan=np.nanmax(b0_mag))

        im = axes[0, 0].imshow(b0_plot.T, origin="lower", extent=ext, aspect="auto", cmap="viridis",
                               vmax=float(np.nanpercentile(b0_mag, 98)))
        axes[0, 0].contour(X * 100, Z * 100, np.nan_to_num(b0_mag), levels=[b_sweet], colors="cyan", linewidths=1.5)
        axes[0, 0].contour(X * 100, Z * 100, np.nan_to_num(b0_mag), levels=[sodium["b_res"]], colors="orange", linewidths=1.5)
        axes[0, 0].axvline(args.borehole_radius_cm, color="w", ls=":", lw=1)
        axes[0, 0].set_title("|B0| (T): 1H sweet spot (cyan), 23Na shell (orange)")
        axes[0, 0].set_xlabel("radius (cm)")
        axes[0, 0].set_ylabel("z (cm)")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

        for ax, res, title in ((axes[0, 1], proton, "1H sensitive region"),
                               (axes[0, 2], sodium, "23Na sensitive region")):
            s = res["sensitivity"]
            vmax = float(np.percentile(s[s > 0], 99)) if np.any(s > 0) else None
            im = ax.imshow(s.T, origin="lower", extent=ext, aspect="auto", cmap="inferno", vmax=vmax)
            ax.set_title(f"{title} (signal density)")
            ax.set_xlabel("radius (cm)")
            ax.set_ylabel("z (cm)")
            fig.colorbar(im, ax=ax, fraction=0.046)

        axes[1, 0].plot(t_acq * 1e3, echo_h * 1e9, label="1H", lw=1.4)
        axes[1, 0].plot(t_acq * 1e3, echo_na * 1e9, label="23Na", lw=1.4)
        axes[1, 0].plot(t_acq * 1e3, (echo_h + echo_na) * 1e9, "k--", lw=1, label="total")
        axes[1, 0].set_title("Received echo through the tuned probe")
        axes[1, 0].set_xlabel("time (ms)")
        axes[1, 0].set_ylabel("echo EMF (nV)")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].bar([0, 1], [proton["total"] * 1e9, sodium["total"] * 1e9], color=["C0", "C1"])
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(["1H", "23Na"])
        axes[1, 1].set_title(f"Signal by nucleus (23Na = {na_fraction * 100:.1f} %)")
        axes[1, 1].set_ylabel("echo EMF (nV)")

        axes[1, 2].axis("off")
        txt = (f"Jasper-Jackson tool\n"
               f"f_RF = {f_rf / 1e6:.3f} MHz\n"
               f"sweet spot r = {r_sweet * 100:.1f} cm\n"
               f"1H |B0| = {b_sweet:.4f} T\n"
               f"23Na |B0| = {sodium['b_res']:.4f} T\n"
               f"23Na flip ~ {na_flip_med:.0f} deg\n\n"
               f"90 / 180 = {t90 * 1e6:.0f} / {t180 * 1e6:.0f} us\n"
               f"peak current = {coil_current:.0f} A\n"
               f"RF power = {rf_power / 1e3:.1f} kW\n\n"
               f"1H echo  = {proton['total'] * 1e9:.2f} nV\n"
               f"23Na echo = {sodium['total'] * 1e9:.2f} nV\n"
               f"23Na fraction = {na_fraction * 100:.1f} %\n\n"
               f"R_total = {r_total:.2f} ohm\n"
               f"noise = {noise_rms * 1e9:.2f} nV\n"
               f"SNR/echo = {snr:.1f}")
        axes[1, 2].text(0.05, 0.95, txt, va="top", ha="left", fontsize=11, family="monospace")
        axes[1, 2].set_title("Summary")
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
