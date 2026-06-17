"""Result export helpers for optimization driver outputs.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_*_repeat.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned_repeat.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned_inv_repeat.m
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatlabResultSummary:
    """Compact score summary extracted from MATLAB-style result cells."""

    pulse_kind: str
    scores: np.ndarray
    secondary_scores: np.ndarray
    best_index: int
    selected_index: int

    @property
    def best_score(self) -> float:
        return float(self.scores[self.best_index])

    @property
    def selected_score(self) -> float:
        return float(self.scores[self.selected_index])


@dataclass(frozen=True)
class PulseProgram:
    """Piecewise-constant pulse program extracted from optimization results."""

    times: np.ndarray
    phases: np.ndarray
    amplitudes: np.ndarray


@dataclass(frozen=True)
class SelectedOptimizationProgram:
    """Selected pulse program and score from MATLAB-style result cells."""

    pulse_kind: str
    pulse_number: int
    score: float
    secondary_score: float | None
    refocusing: PulseProgram | None
    excitation: PulseProgram | None


@dataclass(frozen=True)
class MatlabResultLayout:
    """Column layout used by a MATLAB `plot_opt_*_results` script."""

    name: str
    matlab_script: str
    pulse_kind: str
    score_index: int
    score_label: str
    params_index: int
    sp_index: int
    pp_index: int
    maximize: bool = True
    secondary_index: int | None = None
    secondary_label: str | None = None


@dataclass(frozen=True)
class OptimizationResultAnalysis:
    """Script-aware, plotting-free analysis of MATLAB optimization result cells."""

    layout: MatlabResultLayout
    summary: MatlabResultSummary
    selected_program: SelectedOptimizationProgram
    params: Any
    sp: Any
    pp: Any

    @property
    def score_label(self) -> str:
        return self.layout.score_label

    @property
    def secondary_label(self) -> str | None:
        return self.layout.secondary_label


@dataclass(frozen=True)
class TunedInverseResultPairAnalysis:
    """Comparison corresponding to `plot_opt_exc_results_tuned_inv.m`."""

    original: OptimizationResultAnalysis
    inverse: OptimizationResultAnalysis

    @property
    def original_score(self) -> float:
        return self.original.summary.selected_score

    @property
    def inverse_score(self) -> float:
        return self.inverse.summary.selected_score

    @property
    def score_difference(self) -> float:
        return self.inverse_score - self.original_score


_LAYOUTS_BY_NAME: dict[str, MatlabResultLayout] = {
    "ideal_v0crit_refocusing": MatlabResultLayout(
        name="ideal_v0crit_refocusing",
        matlab_script="plot_opt_ref_results_ideal_v0crit.m",
        pulse_kind="refocusing",
        score_index=3,
        score_label="Optimized SNR",
        secondary_index=4,
        secondary_label="Average v0crit",
        params_index=5,
        sp_index=6,
        pp_index=7,
    ),
    "tuned_refocusing": MatlabResultLayout(
        name="tuned_refocusing",
        matlab_script="plot_opt_ref_results_tuned.m",
        pulse_kind="refocusing",
        score_index=3,
        score_label="Optimized SNR (rms)",
        params_index=4,
        sp_index=5,
        pp_index=6,
    ),
    "untuned_refocusing": MatlabResultLayout(
        name="untuned_refocusing",
        matlab_script="plot_opt_ref_results_untuned.m",
        pulse_kind="refocusing",
        score_index=3,
        score_label="Optimized SNR (rms)",
        params_index=4,
        sp_index=5,
        pp_index=6,
    ),
    "matched_refocusing": MatlabResultLayout(
        name="matched_refocusing",
        matlab_script="plot_opt_ref_results_matched.m",
        pulse_kind="refocusing",
        score_index=3,
        score_label="Optimized SNR (rms)",
        params_index=4,
        sp_index=5,
        pp_index=6,
    ),
    "ideal_time_varying_refocusing": MatlabResultLayout(
        name="ideal_time_varying_refocusing",
        matlab_script="plot_opt_ref_results_ideal_tv.m",
        pulse_kind="refocusing",
        score_index=3,
        score_label="Optimized SNR",
        params_index=4,
        sp_index=5,
        pp_index=6,
    ),
    "tuned_excitation": MatlabResultLayout(
        name="tuned_excitation",
        matlab_script="plot_opt_exc_results_tuned.m",
        pulse_kind="excitation",
        score_index=6,
        score_label="Optimized SNR (rms)",
        params_index=7,
        sp_index=8,
        pp_index=9,
    ),
    "tuned_inverse_excitation": MatlabResultLayout(
        name="tuned_inverse_excitation",
        matlab_script="plot_opt_exc_results_tuned_inv.m",
        pulse_kind="inverse_excitation",
        score_index=6,
        score_label="Optimized inverse mismatch",
        params_index=7,
        sp_index=8,
        pp_index=9,
        maximize=False,
    ),
}

_LAYOUT_ALIASES = {
    layout.name: layout.name for layout in _LAYOUTS_BY_NAME.values()
}
_LAYOUT_ALIASES.update(
    {
        "plot_opt_ref_results_ideal_v0crit": "ideal_v0crit_refocusing",
        "plot_opt_ref_results_ideal_v0crit.m": "ideal_v0crit_refocusing",
        "ideal_v0crit": "ideal_v0crit_refocusing",
        "plot_opt_ref_results_tuned": "tuned_refocusing",
        "plot_opt_ref_results_tuned.m": "tuned_refocusing",
        "tuned": "tuned_refocusing",
        "plot_opt_ref_results_untuned": "untuned_refocusing",
        "plot_opt_ref_results_untuned.m": "untuned_refocusing",
        "untuned": "untuned_refocusing",
        "plot_opt_ref_results_matched": "matched_refocusing",
        "plot_opt_ref_results_matched.m": "matched_refocusing",
        "matched": "matched_refocusing",
        "plot_opt_ref_results_ideal_tv": "ideal_time_varying_refocusing",
        "plot_opt_ref_results_ideal_tv.m": "ideal_time_varying_refocusing",
        "ideal_tv": "ideal_time_varying_refocusing",
        "ideal_time_varying": "ideal_time_varying_refocusing",
        "plot_opt_exc_results_tuned": "tuned_excitation",
        "plot_opt_exc_results_tuned.m": "tuned_excitation",
        "excitation": "tuned_excitation",
        "plot_opt_exc_results_tuned_inv": "tuned_inverse_excitation",
        "plot_opt_exc_results_tuned_inv.m": "tuned_inverse_excitation",
        "inverse_excitation": "tuned_inverse_excitation",
    }
)


def _as_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _as_scalar_float(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar numeric result cell")
    return float(np.ravel(array)[0])


def _result_phases(result: Any) -> np.ndarray:
    return _as_array(getattr(result, "best_phases"))


def _result_score(result: Any) -> float:
    return float(getattr(result, "best_score"))


def _secondary_refocusing_score(result: Any) -> float | None:
    evaluation = getattr(result, "best_evaluation", None)
    value = getattr(evaluation, "v0crit_average", None)
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def _segment_fraction(result: Any, default: float) -> float:
    evaluation = getattr(result, "best_evaluation", None)
    phases = _result_phases(result)
    pulse_length = getattr(evaluation, "pulse_length_t180", None)
    if pulse_length is None or phases.size == 0:
        return float(default)
    return float(pulse_length) / phases.size


def _summary_struct(multistart: Any, result: Any, index: int) -> dict[str, Any]:
    return {
        "pulse_kind": getattr(multistart, "pulse_kind"),
        "probe": getattr(multistart, "probe"),
        "run_index": int(index + 1),
        "best_index": int(getattr(multistart, "best_index")) + 1,
        "best_score": _result_score(result),
        "optimizer_method": getattr(result, "optimizer_method", ""),
        "optimizer_success": bool(getattr(result, "optimizer_success", True)),
        "optimizer_message": getattr(result, "optimizer_message", ""),
    }


def _refocusing_cell(
    multistart: Any,
    result: Any,
    index: int,
    *,
    segment_fraction_t180: float,
    free_precession_t180: float,
) -> np.ndarray:
    phases = _result_phases(result)
    segment_lengths = float(segment_fraction_t180) * np.ones(phases.size)
    secondary_score = _secondary_refocusing_score(result)
    cell_width = 8 if secondary_score is not None else 7
    cell = np.empty((1, cell_width), dtype=object)
    cell[0, 0] = np.concatenate(
        [[free_precession_t180], segment_lengths, [free_precession_t180]]
    )
    cell[0, 1] = np.concatenate([[0.0], phases, [0.0]])
    cell[0, 2] = np.concatenate([[0.0], np.ones(phases.size), [0.0]])
    cell[0, 3] = _result_score(result)
    metadata_offset = 5 if secondary_score is not None else 4
    if secondary_score is not None:
        cell[0, 4] = secondary_score
    cell[0, metadata_offset] = _summary_struct(multistart, result, index)
    cell[0, metadata_offset + 1] = {
        "initial_phases": _as_array(getattr(result, "initial_phases", phases))
    }
    cell[0, metadata_offset + 2] = {
        "bounds": _as_array(getattr(result, "bounds", getattr(multistart, "bounds"))),
        "history_scores": _as_array(getattr(result, "history_scores", [])),
    }
    return cell


def _excitation_cell(
    multistart: Any,
    result: Any,
    index: int,
    *,
    excitation_segment_fraction_t180: float,
    refocusing_segment_fraction_t180: float,
    free_precession_t180: float,
) -> np.ndarray:
    phases = _result_phases(result)
    excitation_lengths = float(excitation_segment_fraction_t180) * np.ones(phases.size)
    ref_phases = _as_array(
        getattr(
            getattr(result, "best_evaluation", None),
            "reference_phases",
            [],
        )
    )
    ref_lengths = float(refocusing_segment_fraction_t180) * np.ones(ref_phases.size)
    cell = np.empty((1, 10), dtype=object)
    cell[0, 0] = excitation_lengths
    cell[0, 1] = phases
    cell[0, 2] = np.ones(phases.size)
    cell[0, 3] = np.concatenate(
        [[free_precession_t180], ref_lengths, [free_precession_t180]]
    )
    cell[0, 4] = np.concatenate([[0.0], ref_phases, [0.0]])
    cell[0, 5] = np.concatenate([[0.0], np.ones(ref_phases.size), [0.0]])
    cell[0, 6] = _result_score(result)
    cell[0, 7] = _summary_struct(multistart, result, index)
    cell[0, 8] = {
        "initial_phases": _as_array(getattr(result, "initial_phases", phases))
    }
    cell[0, 9] = {
        "bounds": _as_array(getattr(result, "bounds", getattr(multistart, "bounds"))),
        "history_scores": _as_array(getattr(result, "history_scores", [])),
    }
    return cell


def multistart_to_matlab_results(
    multistart: Any,
    *,
    segment_fraction_t180: float | None = None,
    free_precession_t180: float = 0.0,
    excitation_segment_fraction_t180: float | None = None,
    refocusing_segment_fraction_t180: float = 0.1,
) -> np.ndarray:
    """Convert a multi-start optimization result to a MATLAB-style cell array.

    The returned object array mirrors the broad shape saved by MATLAB repeat
    scripts: refocusing searches produce an `N x 1` cell array of `1 x 7`
    cells, v0crit-style refocusing searches include the extra average-v0crit
    metric as `1 x 8`, and excitation/inverse-excitation searches produce
    `1 x 10` cells. Times are stored in T180-normalized units unless the caller
    supplies a different convention through the segment/free-precession
    arguments.
    """

    results = tuple(getattr(multistart, "results"))
    if not results:
        raise ValueError("multistart results must not be empty")
    pulse_kind = getattr(multistart, "pulse_kind")
    out = np.empty((len(results), 1), dtype=object)
    for index, result in enumerate(results):
        if pulse_kind == "refocusing":
            seg = (
                _segment_fraction(result, 0.1)
                if segment_fraction_t180 is None
                else float(segment_fraction_t180)
            )
            out[index, 0] = _refocusing_cell(
                multistart,
                result,
                index,
                segment_fraction_t180=seg,
                free_precession_t180=float(free_precession_t180),
            )
        elif pulse_kind in {"excitation", "inverse_excitation"}:
            seg = (
                _segment_fraction(result, 0.1)
                if excitation_segment_fraction_t180 is None
                else float(excitation_segment_fraction_t180)
            )
            out[index, 0] = _excitation_cell(
                multistart,
                result,
                index,
                excitation_segment_fraction_t180=seg,
                refocusing_segment_fraction_t180=float(
                    refocusing_segment_fraction_t180
                ),
                free_precession_t180=float(free_precession_t180),
            )
        else:
            raise ValueError("unsupported pulse_kind for MATLAB-style export")
    return out


def multistart_summary_arrays(multistart: Any) -> dict[str, np.ndarray]:
    """Return compact numeric arrays useful for non-MATLAB result inspection."""

    results = tuple(getattr(multistart, "results"))
    scores = np.asarray([_result_score(result) for result in results], dtype=np.float64)
    phases = np.vstack([_result_phases(result) for result in results])
    starts = np.asarray(getattr(multistart, "initial_phases"), dtype=np.float64)
    return {
        "scores": scores,
        "best_index": np.asarray(
            [int(getattr(multistart, "best_index"))],
            dtype=np.int64,
        ),
        "best_score": np.asarray(
            [float(getattr(multistart, "best_score"))],
            dtype=np.float64,
        ),
        "best_phases": _result_phases(getattr(multistart, "best_result")),
        "initial_phases": starts,
        "result_phases": phases,
    }


def save_multistart_results_npz(
    multistart: Any,
    path: str | Path,
    *,
    variable_name: str = "results",
    **conversion_options: Any,
) -> Path:
    """Save multi-start results as a NumPy archive with MATLAB-style cells."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matlab_results = multistart_to_matlab_results(multistart, **conversion_options)
    arrays = multistart_summary_arrays(multistart)
    np.savez(
        output_path,
        **{
            variable_name: matlab_results,
            "pulse_kind": np.asarray([getattr(multistart, "pulse_kind")]),
            "probe": np.asarray([getattr(multistart, "probe")]),
            **arrays,
        },
    )
    return output_path


def load_multistart_results_npz(
    path: str | Path,
    *,
    variable_name: str = "results",
) -> dict[str, np.ndarray]:
    """Load a NumPy optimization archive written by `save_multistart_results_npz`."""

    with np.load(Path(path), allow_pickle=True) as data:
        if variable_name not in data.files:
            raise KeyError(f"{variable_name!r} not found in optimization archive")
        return {name: data[name] for name in data.files}


def load_optimization_results(
    path: str | Path,
    *,
    variable_name: str = "results",
) -> np.ndarray:
    """Load optimization result cells from a `.npz` or MATLAB `.mat` file."""

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        archive = load_multistart_results_npz(
            input_path,
            variable_name=variable_name,
        )
        return np.asarray(archive[variable_name], dtype=object)
    if suffix == ".mat":
        return load_matlab_results_mat(input_path, variable_name=variable_name)
    raise ValueError("optimization result path must end with .npz or .mat")


def save_multistart_results_mat(
    multistart: Any,
    path: str | Path,
    *,
    variable_name: str = "results",
    **conversion_options: Any,
) -> Path:
    """Save multi-start results to a MATLAB `.mat` file when SciPy is present."""

    try:
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "SciPy is required for MATLAB .mat export. Install the optional "
            "optimization dependencies with `python -m pip install -e .[opt]`."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matlab_results = multistart_to_matlab_results(multistart, **conversion_options)
    payload = {
        variable_name: matlab_results,
        "summary": multistart_summary_arrays(multistart),
        "pulse_kind": getattr(multistart, "pulse_kind"),
        "probe": getattr(multistart, "probe"),
    }
    savemat(output_path, payload, do_compression=True)
    return output_path


def load_matlab_results_mat(
    path: str | Path,
    *,
    variable_name: str = "results",
) -> np.ndarray:
    """Load MATLAB optimization result cells from a `.mat` file when SciPy exists."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "SciPy is required for MATLAB .mat import. Install the optional "
            "optimization dependencies with `python -m pip install -e .[opt]`."
        ) from exc

    payload = loadmat(Path(path), squeeze_me=False, struct_as_record=False)
    if variable_name not in payload:
        raise KeyError(f"{variable_name!r} not found in MATLAB result file")
    return np.asarray(payload[variable_name], dtype=object)


def _flatten_matlab_results(results: Any) -> tuple[np.ndarray, ...]:
    arr = np.asarray(results, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 1:
        raw_cells = [arr[index, 0] for index in range(arr.shape[0])]
    elif arr.ndim == 1:
        raw_cells = [arr[index] for index in range(arr.shape[0])]
    else:
        raise ValueError("results must be an N x 1 or length-N cell array")

    cells = []
    for raw in raw_cells:
        cell = np.asarray(raw, dtype=object)
        if cell.ndim == 2 and cell.shape[0] == 1:
            cell = cell[0, :]
        elif cell.ndim != 1:
            raise ValueError("each optimization result cell must be 1-D or 1 x M")
        cells.append(cell)
    if not cells:
        raise ValueError("results must not be empty")
    return tuple(cells)


def _infer_pulse_kind(cells: tuple[np.ndarray, ...], pulse_kind: str | None) -> str:
    if pulse_kind is not None:
        if pulse_kind not in {"refocusing", "excitation", "inverse_excitation"}:
            raise ValueError("unsupported pulse_kind")
        return pulse_kind
    width = int(cells[0].size)
    if width == 10:
        return "excitation"
    if width in {7, 8}:
        return "refocusing"
    raise ValueError("cannot infer pulse_kind from MATLAB result cell width")


def matlab_result_layouts() -> dict[str, MatlabResultLayout]:
    """Return known MATLAB optimization result layouts keyed by canonical name."""

    return dict(_LAYOUTS_BY_NAME)


def _normalize_layout_name(name: str) -> str:
    key = name.strip()
    if key.endswith(".m"):
        key = key[:-2] + ".m"
    canonical = _LAYOUT_ALIASES.get(key, _LAYOUT_ALIASES.get(key.lower()))
    if canonical is None:
        raise ValueError(f"unknown MATLAB optimization result layout: {name!r}")
    return canonical


def get_matlab_result_layout(
    layout: str | MatlabResultLayout | None = None,
    *,
    results: Any | None = None,
) -> MatlabResultLayout:
    """Resolve or infer a MATLAB optimization result-cell layout."""

    if isinstance(layout, MatlabResultLayout):
        return layout
    if layout is not None:
        return _LAYOUTS_BY_NAME[_normalize_layout_name(layout)]
    if results is None:
        raise ValueError("layout must be supplied when results are not available")
    cells = _flatten_matlab_results(results)
    width = int(cells[0].size)
    if width == 8:
        return _LAYOUTS_BY_NAME["ideal_v0crit_refocusing"]
    if width == 7:
        return _LAYOUTS_BY_NAME["tuned_refocusing"]
    if width == 10:
        return _LAYOUTS_BY_NAME["tuned_excitation"]
    raise ValueError("cannot infer MATLAB result layout from result cell width")


def _cell_score(cell: np.ndarray, pulse_kind: str) -> float:
    if pulse_kind in {"excitation", "inverse_excitation"}:
        return _as_scalar_float(cell[6])
    return _as_scalar_float(cell[3])


def _cell_secondary_score(cell: np.ndarray, pulse_kind: str) -> float:
    if pulse_kind != "refocusing" or cell.size < 8:
        return np.nan
    value = cell[4]
    try:
        value_float = _as_scalar_float(value)
    except (TypeError, ValueError):
        return np.nan
    return value_float if np.isfinite(value_float) else np.nan


def _cell_value(cell: np.ndarray, index: int, label: str) -> Any:
    if index < 0 or index >= cell.size:
        raise ValueError(f"result cell does not include {label}")
    return cell[index]


def summarize_matlab_results(
    results: Any,
    *,
    pulse_kind: str | None = None,
    pulse_number: int | None = None,
    maximize: bool = True,
) -> MatlabResultSummary:
    """Summarize scores from MATLAB-style optimization result cells.

    `pulse_number` follows the MATLAB scripts and is one-based. When omitted,
    the selected index is the best score.
    """

    cells = _flatten_matlab_results(results)
    kind = _infer_pulse_kind(cells, pulse_kind)
    scores = np.asarray([_cell_score(cell, kind) for cell in cells], dtype=np.float64)
    secondary = np.asarray(
        [_cell_secondary_score(cell, kind) for cell in cells],
        dtype=np.float64,
    )
    best_index = int(np.nanargmax(scores) if maximize else np.nanargmin(scores))
    if pulse_number is None:
        selected_index = best_index
    else:
        selected_index = int(pulse_number) - 1
        if selected_index < 0 or selected_index >= scores.size:
            raise ValueError("pulse_number is outside the available result range")
    return MatlabResultSummary(
        pulse_kind=kind,
        scores=scores,
        secondary_scores=secondary,
        best_index=best_index,
        selected_index=selected_index,
    )


def select_matlab_result_program(
    results: Any,
    *,
    pulse_kind: str | None = None,
    pulse_number: int | None = None,
) -> SelectedOptimizationProgram:
    """Extract the selected pulse program from MATLAB-style result cells.

    This is the plotting-free core of the MATLAB `plot_opt_*_results` scripts:
    choose a one-based pulse number or the best score, then return the stored
    piecewise-constant pulse arrays.
    """

    cells = _flatten_matlab_results(results)
    summary = summarize_matlab_results(
        results,
        pulse_kind=pulse_kind,
        pulse_number=pulse_number,
    )
    cell = cells[summary.selected_index]
    if summary.pulse_kind in {"excitation", "inverse_excitation"}:
        excitation = PulseProgram(
            times=_as_array(cell[0]),
            phases=_as_array(cell[1]),
            amplitudes=_as_array(cell[2]),
        )
        refocusing = PulseProgram(
            times=_as_array(cell[3]),
            phases=_as_array(cell[4]),
            amplitudes=_as_array(cell[5]),
        )
    else:
        excitation = None
        refocusing = PulseProgram(
            times=_as_array(cell[0]),
            phases=_as_array(cell[1]),
            amplitudes=_as_array(cell[2]),
        )
    secondary = summary.secondary_scores[summary.selected_index]
    return SelectedOptimizationProgram(
        pulse_kind=summary.pulse_kind,
        pulse_number=summary.selected_index + 1,
        score=summary.selected_score,
        secondary_score=None if np.isnan(secondary) else float(secondary),
        refocusing=refocusing,
        excitation=excitation,
    )


def analyze_matlab_optimization_results(
    results: Any,
    *,
    layout: str | MatlabResultLayout | None = None,
    pulse_number: int | None = None,
) -> OptimizationResultAnalysis:
    """Analyze MATLAB optimization cells using a specific plot-script layout.

    This is the non-plotting equivalent of the `plot_opt_*_results.m` scripts:
    it selects the requested one-based pulse number, reports the score arrays,
    extracts the pulse program, and returns the corresponding `params`, `sp`,
    and `pp` entries from the selected result cell.
    """

    cells = _flatten_matlab_results(results)
    resolved_layout = get_matlab_result_layout(layout, results=results)
    summary = summarize_matlab_results(
        results,
        pulse_kind=resolved_layout.pulse_kind,
        pulse_number=pulse_number,
        maximize=resolved_layout.maximize,
    )
    selected = select_matlab_result_program(
        results,
        pulse_kind=resolved_layout.pulse_kind,
        pulse_number=summary.selected_index + 1,
    )
    cell = cells[summary.selected_index]
    return OptimizationResultAnalysis(
        layout=resolved_layout,
        summary=summary,
        selected_program=selected,
        params=_cell_value(cell, resolved_layout.params_index, "params"),
        sp=_cell_value(cell, resolved_layout.sp_index, "sp"),
        pp=_cell_value(cell, resolved_layout.pp_index, "pp"),
    )


def analyze_optimization_result_file(
    path: str | Path,
    *,
    layout: str | MatlabResultLayout | None = None,
    pulse_number: int | None = None,
    variable_name: str = "results",
) -> OptimizationResultAnalysis:
    """Load and analyze a `.mat` or `.npz` optimization result file."""

    results = load_optimization_results(path, variable_name=variable_name)
    return analyze_matlab_optimization_results(
        results,
        layout=layout,
        pulse_number=pulse_number,
    )


def analyze_tuned_inverse_result_pair(
    original_results: Any,
    inverse_results: Any,
    *,
    pulse_number: int | None = None,
) -> TunedInverseResultPairAnalysis:
    """Analyze the original/inverse files used by `plot_opt_exc_results_tuned_inv`.

    The MATLAB script calls `plot_opt_exc_results_tuned(file, pulse_num)` and
    then repeats the same extraction for `[file '_inv']`. This helper keeps the
    same selected pulse number for both result sets and exposes the two scores
    for downstream cancellation diagnostics.
    """

    original = analyze_matlab_optimization_results(
        original_results,
        layout="tuned_excitation",
        pulse_number=pulse_number,
    )
    inverse = analyze_matlab_optimization_results(
        inverse_results,
        layout="tuned_inverse_excitation",
        pulse_number=pulse_number,
    )
    return TunedInverseResultPairAnalysis(original=original, inverse=inverse)


def analyze_tuned_inverse_result_files(
    original_path: str | Path,
    inverse_path: str | Path | None = None,
    *,
    pulse_number: int | None = None,
    variable_name: str = "results",
) -> TunedInverseResultPairAnalysis:
    """Load and analyze original/inverse tuned-excitation result files."""

    original = load_optimization_results(original_path, variable_name=variable_name)
    if inverse_path is None:
        original_file = Path(original_path)
        inverse_file = original_file.with_name(
            f"{original_file.stem}_inv{original_file.suffix}"
        )
    else:
        inverse_file = Path(inverse_path)
    inverse = load_optimization_results(inverse_file, variable_name=variable_name)
    return analyze_tuned_inverse_result_pair(
        original,
        inverse,
        pulse_number=pulse_number,
    )
