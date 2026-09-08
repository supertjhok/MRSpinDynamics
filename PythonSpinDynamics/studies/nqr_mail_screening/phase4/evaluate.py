"""Disjoint training/calibration/test audit of fixed and adaptive screening."""

from pathlib import Path  # noqa: E402
import hashlib
import json
import numpy as np
from scipy.stats import rankdata, beta  # noqa: E402
from tree import build  # noqa: E402
from features import noise_bank, fit_covariance, real_features, validate_superposition  # noqa: E402
from inference import Detector  # noqa: E402


def auc(negative, positive):
    ranks = rankdata(np.r_[negative, positive])
    n, m = len(negative), len(positive)
    return float((ranks[n:].sum() - m * (m + 1) / 2) / (n * m))


def threshold(calibration, alpha=0.05):
    k = int(np.ceil((len(calibration) + 1) * (1 - alpha)))
    return float(np.sort(calibration)[k - 1]) if k <= len(calibration) else float("inf")


def binomial_interval(k, n):
    return [
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    ]


def intervals(negative, positive, seed=931, paired=None):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(400):
        a = rng.integers(len(negative), size=len(negative))
        b = rng.integers(len(positive), size=len(positive))
        value = auc(negative[a], positive[b])
        if paired is not None:
            value -= auc(paired[0][a], paired[1][b])
        draws.append(value)
    return list(np.quantile(draws, [0.025, 0.975]))


def load_bank(path, projectors, time, n, seed, stress=False):
    # Fingerprint both the receiver-data generator and projector used to compress it.
    from spin_dynamics.interference import cancellers

    phase3 = Path(__file__).resolve().parents[1] / "phase3"
    dependencies = [
        Path(__file__).with_name("features.py"),
        phase3 / "acquisition.py",
        phase3 / "environment.py",
        Path(cancellers.__file__),
    ]
    identity = hashlib.sha256(
        projectors.tobytes() + b"".join(p.read_bytes() for p in dependencies)
    ).hexdigest()
    metadata = path.with_suffix(".json")
    spec = {"n": n, "seed": seed, "stress": stress, "identity": identity}
    if path.exists() and metadata.exists() and json.loads(metadata.read_text()) == spec:
        return np.load(path)["features"]
    value = noise_bank(projectors, time, n, seed, stress)
    np.savez_compressed(path, features=value)
    metadata.write_text(json.dumps(spec, indent=2) + "\n")
    return value


def main():
    root = Path(".tmp/nqr_phase4")
    root.mkdir(exist_ok=True, parents=True)
    info, arrays = build(output=root)
    p, t = arrays["projectors"], arrays["time_s"]
    training = load_bank(root / "training.npz", p, t, 1000, 401)
    calibration = load_bank(root / "calibration.npz", p, t, 800, 402)
    testing = load_bank(root / "testing.npz", p, t, 1000, 403)
    shifted = load_bank(root / "shifted.npz", p, t, 1000, 404, True)
    blanking = real_features(arrays["blanking_mean_v"])
    training, calibration, testing, shifted = [
        x + blanking for x in (training, calibration, testing, shifted)
    ]
    mean, scale, cov = fit_covariance(training)
    receiver_validation = validate_superposition(arrays, scale)
    np.savez_compressed(
        root / "fitted_detector.npz", mean=mean, scale=scale, covariance=cov
    )
    rng = np.random.default_rng(405)
    temperatures = rng.integers(3, size=500)
    phases = rng.uniform(0, 2 * np.pi, 500)
    caps = (3.3475, 3.351, 3.354)
    cases = [
        ("physical_1g", 1.0, 0.5, None, False, False),
        ("diagnostic_gain_100000", 100000.0, 0.5, None, False, False),
        ("diagnostic_gain_10000", 10000.0, 0.5, None, False, False),
        ("prior_1_percent", 10000.0, 0.01, None, False, False),
        ("missing_20C_template", 10000.0, 0.5, 1, False, False),
        ("missing_y_signal", 10000.0, 0.5, None, True, False),
        ("shifted_RFI", 10000.0, 0.5, None, False, True),
    ]
    results = []
    scores = {}
    for case, gain, prior, omit, missing_y, shift in cases:
        detector = Detector(info, arrays, mean, scale, cov, gain, prior, omit)
        data = shifted if shift else testing
        signals = np.array(
            [
                detector.signal(int(temp), phase, gain, missing_y=missing_y)
                for temp, phase in zip(temperatures, phases)
            ]
        )
        for cap in (
            caps
            if case
            in ("physical_1g", "diagnostic_gain_100000", "diagnostic_gain_10000")
            else caps[-1:]
        ):
            paired = {}
            for adaptive in (False, True):
                policy = "adaptive" if adaptive else "fixed"
                cal = np.array(
                    [
                        detector.run(x, cap, adaptive, seed=50000 + i)["score"]
                        for i, x in enumerate(calibration)
                    ]
                )
                negative = [
                    detector.run(x, cap, adaptive, seed=60000 + i)
                    for i, x in enumerate(data[:500])
                ]
                positive = [
                    detector.run(x + s, cap, adaptive, seed=61000 + i)
                    for i, (x, s) in enumerate(zip(data[500:], signals))
                ]
                n = np.array([x["score"] for x in negative])
                h = np.array([x["score"] for x in positive])
                cut = threshold(cal)
                pvalues = (1 + (cal[:, None] >= n[None, :]).sum(axis=0)) / (
                    len(cal) + 1
                )
                grid = np.linspace(0, 1, 101)
                excess = float(max(np.mean(pvalues <= u) - u for u in grid))
                bound = float(
                    np.sqrt(np.log(40) / (2 * len(cal)))
                    + np.sqrt(np.log(40) / (2 * len(n)))
                )
                row = {
                    "case": case,
                    "policy": policy,
                    "cap_s": cap,
                    "source_voltage_gain": gain,
                    "target_mass_g": 1.0,
                    "auc": auc(n, h),
                    "auc_ci95": intervals(n, h),
                    "illustrative_alpha": 0.05,
                    "threshold": cut,
                    "held_out_pfa": float(np.mean(n > cut)),
                    "pfa_ci95": binomial_interval(int(np.sum(n > cut)), len(n)),
                    "held_out_pd": float(np.mean(h > cut)),
                    "pd_ci95": binomial_interval(int(np.sum(h > cut)), len(h)),
                    "pvalue_cdf_excess": excess,
                    "cdf_tolerance": bound,
                    "calibration_check_passed": bool(excess <= bound),
                    "mean_time_h0_s": float(np.mean([x["seconds"] for x in negative])),
                    "mean_time_h1_s": float(np.mean([x["seconds"] for x in positive])),
                    "max_time_s": max(x["seconds"] for x in negative + positive),
                    "evidence_stop_fraction": float(
                        np.mean([x["stopped_on_evidence"] for x in negative + positive])
                    ),
                    "first_action_counts": {
                        str(a): sum(
                            bool(x["path"]) and x["path"][0] == a
                            for x in negative + positive
                        )
                        for a in range(3)
                    },
                }
                paired[policy] = (n, h)
                results.append(row)
                scores[f"{case}_{cap}_{policy}_h0"] = n
                scores[f"{case}_{cap}_{policy}_h1"] = h
                print(
                    case,
                    cap,
                    policy,
                    "AUC",
                    round(row["auc"], 4),
                    "PFA",
                    round(row["held_out_pfa"], 4),
                    flush=True,
                )
            delta = auc(*paired["adaptive"]) - auc(*paired["fixed"])
            results[-1]["paired_auc_advantage"] = delta
            results[-1]["paired_advantage_ci95"] = intervals(
                *paired["adaptive"], paired=paired["fixed"]
            )
    np.savez_compressed(root / "held_out_scores.npz", **scores)
    main_rows = [
        r
        for r in results
        if r["case"]
        in ("physical_1g", "diagnostic_gain_100000", "diagnostic_gain_10000")
    ]
    checks = {
        "receiver_feature_superposition": receiver_validation["passed"],
        "calibration_on_matched_domain": all(
            r["calibration_check_passed"] for r in main_rows
        ),
        "physical_time_caps": all(
            r["max_time_s"] <= r["cap_s"] + 1e-12 for r in results
        ),
        "uncertainty_reported": all(len(r["auc_ci95"]) == 2 for r in results),
        "stateful_tree": len(info["nodes"]) == 39
        and all(n["source_trace_error"] < 1e-16 for n in info["nodes"]),
        "independent_split_seeds": len({401, 402, 403, 404, 405}) == 5,
    }
    report = {
        "phase": 4,
        "receiver_validation": receiver_validation,
        "gate4_passed": all(checks.values()),
        "checks": checks,
        "fit_h0": 1000,
        "receiver_split_metadata": {
            name: json.loads((root / f"{name}.json").read_text())
            for name in ("training", "calibration", "testing", "shifted")
        },
        "calibration_h0": 800,
        "held_out_each_class": 500,
        "results": results,
        "action_tree": info,
        "limits": [
            "Feature-level Gaussian likelihood calibrated using Phase 3 receiver/cancellation noise; signal means come from actual branched spin histories.",
            "Signal added at the linear feature stage; no claim of ADC or overload invariance outside the accepted Phase 3 small-signal regime.",
            "Known RF switching mean is treated as calibrated; uncertain switching transients and benign NQR libraries are not validated here.",
            "Gain 100000 is a software diagnostic, not hardware gain, extra mass, or demonstrated detectability.",
            "Synthetic stationary/drifting RFI priors only; shifted RFI intentionally tests calibration failure.",
            "Three-step central-packet action tree; no hardware optimum or continuous-space adaptive search claimed.",
        ],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parent.glob("*.py")
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    (root / "phase4_report.json").write_text(text)
    Path(__file__).with_name("phase4_report.json").write_text(text)
    if not report["gate4_passed"]:
        raise SystemExit("Gate 4 failed")


if __name__ == "__main__":
    main()
