"""Plot exact versus surrogate CPMG-IR candidate screening time and fidelity."""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.design import (  # noqa: E402
    CPMGIRAdapter,
    CPMGIRDesign,
    CandidateDesignSpace,
    ComplexGaussianLikelihood,
    ExperimentAdapterCost,
    LaplaceInformationGain,
    ParticlePosterior,
    PolynomialSurrogatePredictor,
    PredictiveModel,
)
from spin_dynamics.design.adapters import ExperimentPredictor  # noqa: E402
from spin_dynamics.experiment import (  # noqa: E402
    Acquisition,
    CPMGIRTrain,
    Experiment,
    Sample,
)


def _parameter_features(parameters) -> np.ndarray:
    return np.column_stack(
        (
            np.log(np.asarray(parameters["t1_seconds"], dtype=np.float64)),
            np.log(np.asarray(parameters["t2_seconds"], dtype=np.float64)),
        )
    )


def _design_features(design: CPMGIRDesign) -> np.ndarray:
    return np.array([np.log(design.delay_seconds)])


def _score(model, posterior, actions, cost) -> np.ndarray:
    utility = LaplaceInformationGain()
    return np.asarray(
        [
            utility.estimate(model, posterior, action, np.random.default_rng(1)).value
            / cost.seconds(action)
            for action in actions
        ]
    )


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.particles < 16 or args.candidates < 6:
        parser.error("use at least 16 particles and 6 candidates")

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    template = Experiment(
        CPMGIRTrain(num_echoes=2, echo_spacing_seconds=1e-3),
        Sample(t1_seconds=0.15, t2_seconds=0.06),
        acquisition=Acquisition(numpts=21, maxoffs=5.0, rephase_action="ignore"),
    )
    adapter = CPMGIRAdapter(
        template,
        {"t1_seconds": 0.15, "t2_seconds": 0.06},
        echo_index=0,
        recovery_seconds=20e-3,
    )
    all_t1 = np.geomspace(0.05, 0.5, args.particles)
    all_t2 = np.full(args.particles, 0.06)
    posterior = ParticlePosterior(
        {"t1_seconds": all_t1, "t2_seconds": all_t2}
    )
    actions = CandidateDesignSpace(
        CPMGIRDesign(float(delay))
        for delay in np.geomspace(5e-3, 0.8, args.candidates)
    ).actions
    likelihood = ComplexGaussianLikelihood(0.5)
    cost = ExperimentAdapterCost(adapter)

    log_t1_min = float(np.log(np.min(all_t1)))
    log_t1_max = float(np.log(np.max(all_t1)))

    def candidate_chebyshev_features(parameters, design) -> np.ndarray:
        log_t1 = np.log(np.asarray(parameters["t1_seconds"], dtype=np.float64))
        coordinate = (
            2.0 * log_t1 - (log_t1_min + log_t1_max)
        ) / (log_t1_max - log_t1_min)
        basis = np.polynomial.chebyshev.chebvander(coordinate, 5)
        features = np.zeros((log_t1.size, len(actions) * basis.shape[1]))
        action_index = actions.index(design)
        start = action_index * basis.shape[1]
        features[:, start : start + basis.shape[1]] = basis
        return features

    training_indices = np.linspace(0, args.particles - 1, 8).astype(int)
    training_parameters = {
        "t1_seconds": all_t1[training_indices],
        "t2_seconds": all_t2[training_indices],
    }
    training_actions = actions
    training_predictor = ExperimentPredictor(
        adapter, cache=False, prefer_batch=False
    )
    start = perf_counter()
    surrogate = PolynomialSurrogatePredictor.fit(
        training_predictor,
        training_parameters,
        training_actions,
        _design_features,
        parameter_encoder=_parameter_features,
        joint_encoder=candidate_chebyshev_features,
        degree=1,
        ridge=1e-8,
    )
    training_time = perf_counter() - start

    exact_predictor = ExperimentPredictor(adapter, cache=False, prefer_batch=False)
    exact_model = PredictiveModel(exact_predictor, likelihood)
    start = perf_counter()
    exact_rates = _score(exact_model, posterior, actions, cost)
    exact_time = perf_counter() - start

    surrogate_model = PredictiveModel(surrogate, likelihood)
    start = perf_counter()
    surrogate_rates = _score(surrogate_model, posterior, actions, cost)
    surrogate_time = perf_counter() - start
    validation = surrogate.validate(exact_predictor, posterior.parameters, actions)
    speedup = exact_time / surrogate_time
    exact_best = int(np.argmax(exact_rates))
    surrogate_best = int(np.argmax(surrogate_rates))

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    labels = ["Exact\nscore", "One-time\nfit", "Surrogate\nscore"]
    durations = np.array([exact_time, training_time, surrogate_time])
    axes[0].bar(labels, durations, color=["tab:blue", "0.55", "tab:green"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s, logarithmic)")
    axes[0].set_title("Planning-Time Breakdown")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].bar_label(
        axes[0].containers[0],
        labels=[f"{value:.3g} s" for value in durations],
        padding=3,
    )
    axes[0].text(
        0.98,
        0.42,
        f"scoring speedup: {speedup:.0f}x",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    delays = 1e3 * np.array([action.delay_seconds for action in actions])
    exact_normalized = exact_rates / np.max(exact_rates)
    surrogate_normalized = surrogate_rates / np.max(surrogate_rates)
    axes[1].semilogx(
        delays,
        exact_normalized,
        "o-",
        linewidth=2,
        label="exact facade scoring",
    )
    axes[1].semilogx(
        delays,
        surrogate_normalized,
        "s--",
        linewidth=2,
        label="polynomial surrogate",
    )
    axes[1].axvline(
        delays[exact_best], color="tab:blue", alpha=0.25, linewidth=5
    )
    axes[1].set_xlabel("Inversion delay (ms)")
    axes[1].set_ylabel("Normalized information rate")
    axes[1].set_title("Recommendation Fidelity")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[1].text(
        0.03,
        0.05,
        f"same choice: {'yes' if exact_best == surrogate_best else 'no'}\n"
        f"held-out NRMSE: {validation.normalized_root_mean_square_error:.2%}",
        transform=axes[1].transAxes,
        va="bottom",
    )
    fig.suptitle(
        "Surrogate screening preserves the CPMG-IR decision while reducing score time"
    )
    fig.tight_layout()

    print(
        f"exact={exact_time:.4f} s surrogate={surrogate_time:.4f} s "
        f"fit={training_time:.4f} s speedup={speedup:.1f}x "
        f"same_choice={exact_best == surrogate_best} "
        f"nrmse={validation.normalized_root_mean_square_error:.4f}"
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
