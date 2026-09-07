"""Render the Phase 2 convergence summary from the completed gate report."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    source = Path(".tmp/nqr_phase2_gate/gate2_report.json")
    report = json.loads(source.read_text())
    names = list(report["candidates"])
    metrics = [
        "field_rms",
        "field_path",
        "peec_surface",
        "voxel_electric",
        "material_surrogate",
    ]
    labels = [
        "Map RMS",
        "Wire field",
        "PEEC surface",
        "Loading integral",
        "Material mesh",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25
    for i, name in enumerate(names):
        data = report["candidates"][name]["checks"]
        ax.bar(
            x + (i - 1) * width,
            [100 * data[m] for m in metrics],
            width,
            label=name.replace("_", " "),
        )
    for i, m in enumerate(metrics):
        ax.hlines(
            100 * report["limits"][m], i - 0.45, i + 0.45, color="black", linestyle="--"
        )
    ax.set_xticks(x, labels)
    ax.set(
        ylabel="Relative discrepancy (%)",
        title="Phase 2 numerical checks; dashed lines are declared limits",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(source.with_name("gate2_convergence.png"), dpi=170)


if __name__ == "__main__":
    main()
