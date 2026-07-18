"""Plot Phase 3 batched-PGSE prediction speedup with exact parity checks."""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.design import PGSEAdapter, PGSEDesign  # noqa: E402
from spin_dynamics.design.adapters import ExperimentPredictor  # noqa: E402
from spin_dynamics.experiment import Experiment, PGSE, Sample  # noqa: E402


def _particles(count: int) -> dict[str, np.ndarray]:
    coordinate = np.linspace(0.0, 1.0, count)
    return {
        "diffusion_coefficient": 0.4e-9 + 1.8e-9 * coordinate,
        "t2_seconds": 0.045 + 0.12 * coordinate[::-1],
    }


def _elapsed(function, repeats: int) -> tuple[float, np.ndarray]:
    best = np.inf
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        best = min(best, perf_counter() - start)
    assert value is not None
    return float(best), np.asarray(value)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-particles", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.maximum_particles < 16 or args.repeats <= 0:
        parser.error("maximum-particles must be at least 16 and repeats positive")

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    template = Experiment(
        PGSE(num_echoes=4),
        Sample(t2_seconds=0.1, diffusion_coefficient=1e-9),
    )
    adapter = PGSEAdapter(
        template,
        {"diffusion_coefficient": 1e-9, "t2_seconds": 0.1},
    )
    action = PGSEDesign(0.06, 2e-3, 18e-3)
    powers = np.arange(4, int(np.floor(np.log2(args.maximum_particles))) + 1, 2)
    counts = np.unique(np.append(2**powers, args.maximum_particles)).astype(int)
    batched_times = []
    facade_times = []

    for count in counts:
        particles = _particles(int(count))
        batched, batched_value = _elapsed(
            lambda: ExperimentPredictor(
                adapter, cache=False, prefer_batch=True
            )(particles, action),
            args.repeats,
        )
        facade, facade_value = _elapsed(
            lambda: ExperimentPredictor(
                adapter, cache=False, prefer_batch=False
            )(particles, action),
            args.repeats,
        )
        if not np.allclose(batched_value, facade_value, rtol=2e-15, atol=2e-15):
            raise RuntimeError("batched and facade PGSE predictions disagree")
        batched_times.append(batched)
        facade_times.append(facade)

    batched_times = np.asarray(batched_times)
    facade_times = np.asarray(facade_times)
    speedup = facade_times / batched_times

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].loglog(
        counts,
        1e3 * facade_times,
        "o-",
        linewidth=2,
        label="one Experiment.run() per particle",
    )
    axes[0].loglog(
        counts,
        1e3 * batched_times,
        "s-",
        linewidth=2,
        label="Phase 3 vectorized equations",
    )
    axes[0].set_xlabel("Posterior particles")
    axes[0].set_ylabel("Prediction wall time (ms, best repeat)")
    axes[0].set_title("Exact PGSE Prediction Cost")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].bar(np.arange(counts.size), speedup, color="tab:green")
    axes[1].set_xticks(np.arange(counts.size), [str(value) for value in counts])
    axes[1].set_xlabel("Posterior particles")
    axes[1].set_ylabel("Speedup (facade / batched)")
    axes[1].set_title("Same Signal, Less Planning Time")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].bar_label(
        axes[1].containers[0],
        labels=[f"{value:.0f}x" for value in speedup],
        padding=3,
    )
    fig.suptitle(
        "Batched deterministic PGSE is numerically identical to facade-per-particle scoring"
    )
    fig.tight_layout()

    print(
        f"largest case: {counts[-1]} particles, "
        f"{facade_times[-1]:.4f} s facade, {batched_times[-1]:.4f} s batched, "
        f"{speedup[-1]:.1f}x speedup"
    )
    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
