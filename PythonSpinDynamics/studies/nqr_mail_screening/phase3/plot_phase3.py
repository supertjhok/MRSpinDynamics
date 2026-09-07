"""Plot actual Phase 3 records, cancellation and reported margins."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(".tmp/nqr_phase3")
    report = json.loads((root / "phase3_report.json").read_text())
    key = "enclosing_helix_293.15K_prepol0"
    data = np.load(root / f"{key}_stationary_multi_reference.npz")
    record = next(r for r in report["records"] if r["id"] == key)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    t = data["time_s"] * 1000
    axes[0].plot(t, abs(data["signal_v"]) * 1e12, color="C0")
    axes[0].set(
        ylabel="1 g source |EMF| (pV)",
        title="Moving packet: retained spin state across x/y-line acquisitions",
    )
    for b in record["blocks"]:
        axes[0].axvline(b["block_start_s"] * 1000, color="0.7", lw=0.7)
        axes[0].text(
            b["block_start_s"] * 1000,
            axes[0].get_ylim()[1] * 0.85,
            b["config"]["sequence"] + "-" + b["config"]["line_id"],
            fontsize=8,
        )
    mask = data["receiver_valid"]
    axes[1].plot(
        t,
        np.where(mask, data["rfi_adc_v"].real * 1e6, np.nan),
        label="RFI + noise",
        alpha=0.6,
    )
    axes[1].plot(
        t,
        np.where(mask, data["cancelled_adc_v"].real * 1e6, np.nan),
        label="Multi-reference residual",
    )
    axes[1].set(ylabel="Input-referred IQ (uV)", xlabel="Physical record time (ms)")
    axes[1].legend(loc="upper right")
    methods = [
        "none",
        "passive_20db",
        "single_reference",
        "multi_reference",
        "protected_adaptive",
    ]
    for i, family in enumerate(("stationary", "nearby_source", "drift")):
        trials = [
            next(
                x
                for x in record["trials"]
                if x["family"] == family and x["method"] == m
            )
            for m in methods
        ]
        axes[2].plot(
            range(5), [x["residual_rms_v"] * 1e6 for x in trials], "o-", label=family
        )
    axes[2].set(
        yscale="log",
        ylabel="Residual contamination RMS (uV)",
        xticks=range(5),
        xticklabels=[
            "None",
            "20 dB shield prior",
            "One reference",
            "Three references",
            "Protected NLMS",
        ],
    )
    axes[2].legend()
    fig.suptitle(
        "Phase 3 synthetic receiver audit — unoptimized pulse candidate, no detection claim",
        fontsize=12,
    )
    target = Path(__file__).resolve().parent / "phase3_records.png"
    fig.savefig(target, dpi=160)
    fig.savefig(root / target.name, dpi=160)
    print(target)


if __name__ == "__main__":
    main()
