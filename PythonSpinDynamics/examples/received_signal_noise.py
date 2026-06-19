"""Demonstrate opt-in received-signal noise models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.noise import NoiseSpec
from spin_dynamics.workflows import (
    run_ideal_cpmg,
    run_ideal_cpmg_imaging,
    run_tuned_cpmg,
    run_tuned_cpmg_imaging,
    summarize_imaging_noise_trials,
)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2)))


def _noise_rms(noisy: np.ndarray, clean: np.ndarray) -> float:
    return _rms(np.asarray(noisy) - np.asarray(clean))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=51, help="Number of CPMG offsets.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path for selected clean/noisy arrays.",
    )
    args = parser.parse_args()

    # Ideal CPMG has no circuit model, so the natural first noise model is
    # additive complex white Gaussian noise. The deterministic arrays remain in
    # `echo` and `mrx`; noisy companions are exposed as `echo_noisy` and
    # `mrx_noisy`.
    ideal = run_ideal_cpmg(
        numpts=args.numpts,
        noise=NoiseSpec(sigma=1e-3, seed=args.seed),
    )

    # Probe-aware workflows can use output-referred receiver noise density.
    # `target_snr` rescales the generated realization to a convenient level for
    # compact examples while still using the probe's frequency-dependent noise
    # shape.
    tuned = run_tuned_cpmg(
        numpts=args.numpts,
        noise=NoiseSpec(model="probe", target_snr=25.0, seed=args.seed),
    )

    # The small phantom is intentionally asymmetric so that k-space and image
    # noise summaries are not dominated by a single repeated pixel value.
    rho = np.array([[1.0, 0.2], [0.4, 0.8]], dtype=np.float64)

    # Imaging white noise is added in k-space before reconstruction. This makes
    # `image_noisy` and `magnitude_noisy` consequences of the same reconstruction
    # path used by the clean result.
    ideal_img = run_ideal_cpmg_imaging(
        rho,
        num_echoes=1,
        ny=3,
        num_workers=1,
        phase_workers=1,
        noise=NoiseSpec(sigma=1e-3, seed=args.seed),
    )

    # Probe-colored imaging noise is available on the physical receiver-output
    # path. For tuned imaging that means `receive_mode="weighted"` rather than
    # the default raw-current MATLAB display convention.
    tuned_img = run_tuned_cpmg_imaging(
        np.ones((1, 1), dtype=np.float64),
        num_echoes=1,
        ny=1,
        num_workers=1,
        phase_workers=1,
        receive_mode="weighted",
        noise=NoiseSpec(model="probe", target_snr=20.0, seed=args.seed),
    )

    # Repeated trials let downstream analysis estimate image-domain noise
    # statistics after reconstruction and image formation. The helper accepts
    # masks so users can separate signal pixels from background pixels.
    imaging_trials = [
        run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
            noise=NoiseSpec(sigma=1e-3, seed=args.seed + idx),
        )
        for idx in range(6)
    ]
    imaging_stats = summarize_imaging_noise_trials(
        imaging_trials,
        signal_mask=rho > 0.3,
        background_mask=rho <= 0.3,
    )

    print("Received signal noise example")
    print(f"num offsets: {ideal.del_w.size}")
    print(f"ideal CPMG clean echo RMS: {_rms(ideal.echo):.12g}")
    print(f"ideal CPMG noise RMS: {_noise_rms(ideal.echo_noisy, ideal.echo):.12g}")
    print(f"ideal CPMG realized SNR: {ideal.noise.realized_snr:.12g}")
    print(f"tuned CPMG probe-noise model: {tuned.noise.model}")
    print(f"tuned CPMG clean echo RMS: {_rms(tuned.echo):.12g}")
    print(f"tuned CPMG noise RMS: {_noise_rms(tuned.echo_noisy, tuned.echo):.12g}")
    print(f"tuned CPMG realized spectral SNR: {tuned.noise.realized_snr:.12g}")
    print(f"ideal imaging k-space shape: {ideal_img.kspace.shape}")
    print(
        "ideal imaging k-space noise RMS: "
        f"{_noise_rms(ideal_img.kspace_noisy, ideal_img.kspace):.12g}"
    )
    print("tuned imaging receive mode: weighted")
    print(
        "tuned imaging probe-noise RMS: "
        f"{_noise_rms(tuned_img.kspace_noisy, tuned_img.kspace):.12g}"
    )
    print(f"imaging trials: {imaging_stats.num_trials}")
    print(f"imaging background noise RMS: {imaging_stats.background_noise_rms:.12g}")
    print(f"imaging signal/background SNR: {imaging_stats.snr:.12g}")

    if args.save_npz is not None:
        # Save both clean and noisy arrays so notebooks can reproduce these
        # summaries or inspect the actual noise realizations.
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=ideal.del_w,
            ideal_echo=ideal.echo,
            ideal_echo_noisy=ideal.echo_noisy,
            tuned_echo=tuned.echo,
            tuned_echo_noisy=tuned.echo_noisy,
            ideal_imaging_kspace=ideal_img.kspace,
            ideal_imaging_kspace_noisy=ideal_img.kspace_noisy,
            tuned_imaging_kspace=tuned_img.kspace,
            tuned_imaging_kspace_noisy=tuned_img.kspace_noisy,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
