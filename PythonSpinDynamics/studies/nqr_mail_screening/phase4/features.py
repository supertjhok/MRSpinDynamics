"""Correlated feature surrogate calibrated with the actual Phase 3 receiver."""

from pathlib import Path  # noqa: E402
import sys
import numpy as np

PHASE3 = Path(__file__).resolve().parents[1] / "phase3"
sys.path.insert(0, str(PHASE3))
from acquisition import Receiver, receiver  # noqa: E402
from environment import rfi_sources  # noqa: E402

sys.path.remove(str(PHASE3))
from spin_dynamics.interference.cancellers import fit_gated_ridge_fir  # noqa: E402


def real_features(z):
    return np.concatenate((z.real, z.imag), axis=-1)


def noise_bank(projectors, time, n, seed, stress=False):
    rng = np.random.default_rng(seed)
    cfg = Receiver()
    fit = (time >= 0.0055) & (time < 0.0099)
    out = []
    psd = 4 * 1.380649e-23 * 293.15 * 7.3 + 1e-18
    sigma = np.sqrt(psd * cfg.fs_hz / 2)
    for trial in range(n):
        case_seed = int(rng.integers(1, 2**31))
        family = "drift" if rng.random() < 0.5 else "stationary"
        rfi, refs = rfi_sources(time, family, case_seed)
        if stress:
            rfi = rfi * (1 + time / time[-1])

        def noise():
            return sigma * (
                rng.normal(size=len(time)) + 1j * rng.normal(size=len(time))
            )

        x = np.array(
            [
                receiver(ref + noise(), np.ones(len(time), bool), cfg)[0] / 2e-6
                for ref in refs
            ]
        )
        y, _, margin = receiver(rfi + noise(), np.ones(len(time), bool), cfg)
        if margin["adc_clipped_samples"]:
            raise ValueError("training record clipped")
        model = fit_gated_ridge_fir(y, x, fit, taps=3, ridge=1e-5)
        cleaned = model.apply(y, x).cleaned
        out.append(real_features(np.einsum("nkt,t->nk", projectors, cleaned)))
        if (trial + 1) % 250 == 0:
            print("receiver bank", seed, trial + 1, "/", n, flush=True)
    return np.array(out)


def fit_covariance(training):
    shape = training.shape[1:]
    flat = training.reshape(len(training), -1)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0, ddof=1)
    z = (flat - mean) / scale
    empirical = np.cov(z, rowvar=False)
    covariance = 0.95 * empirical + 0.05 * np.eye(len(mean))
    return mean.reshape(shape), scale.reshape(shape), covariance


def validate_superposition(arrays, noise_scale):
    from tree import filtered

    rng = np.random.default_rng(432)
    errors = []
    clipped = 0
    for node in range(len(arrays["projectors"])):
        for gain in (1.0, 100000.0):
            noise = 3e-7 * (rng.normal(size=5000) + 1j * rng.normal(size=5000))
            added = arrays["source_v"][1, node] * gain + arrays["leakage_input_v"][node]
            a, _, margin = receiver(noise + added, np.ones(5000, bool))
            b, _, _ = receiver(noise, np.ones(5000, bool))
            error = real_features(
                arrays["projectors"][node] @ (a - b - filtered(added))
            )
            errors.append(float(np.linalg.norm(error / noise_scale[node]) / 2))
            clipped += margin["adc_clipped_samples"]
    return {
        "cases": len(errors),
        "max_feature_error_in_noise_sigma": max(errors),
        "clipped_samples": clipped,
        "passed": bool(max(errors) < 0.05 and clipped == 0),
    }
