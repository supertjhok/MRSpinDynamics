"""Held-out ROC and time comparisons; diagnostic gain is visibly separated."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def roc(a, b):
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[a, b]))[::-1], -np.inf]
    return [np.mean(a > x) for x in thresholds], [np.mean(b > x) for x in thresholds]


def main():
    root = Path(".tmp/nqr_phase4")
    report = json.loads((root / "phase4_report.json").read_text())
    scores = np.load(root / "held_out_scores.npz")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for column, case in enumerate(("physical_1g", "diagnostic_gain_10000")):
        for policy in ("fixed", "adaptive"):
            key = f"{case}_3.354_{policy}"
            fpr, tpr = roc(scores[key + "_h0"], scores[key + "_h1"])
            rows = [
                r
                for r in report["results"]
                if r["case"] == case and r["policy"] == policy
            ]
            row = rows[-1]
            axes[0, column].plot(fpr, tpr, label=f"{policy}: AUC {row['auc']:.3f}")
            x = np.array([r["cap_s"] for r in rows])
            y = np.array([r["auc"] for r in rows])
            lo = np.array([r["auc_ci95"][0] for r in rows])
            hi = np.array([r["auc_ci95"][1] for r in rows])
            axes[1, column].errorbar(
                x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, label=policy
            )
        axes[0, column].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0, column].set(
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            title="Physical 1 g reference" if column == 0 else "",
        )
        axes[0, column].legend()
        axes[1, column].set(
            xlabel="Full parcel time budget (s)",
            ylabel="Held-out AUC, 95% bootstrap interval",
        )
        axes[1, column].legend()
    axes[0, 1].set_title("Software diagnostic: source amplitude x10000")
    fig.suptitle(
        "Phase 4 — synthetic held-out evaluation; no hardware sensitivity claim"
    )
    fig.savefig(Path(__file__).with_name("phase4_results.png"), dpi=160)


if __name__ == "__main__":
    main()
