"""Result export helpers for optimization driver outputs.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_*_repeat.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned_repeat.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned_inv_repeat.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _as_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _result_phases(result: Any) -> np.ndarray:
    return _as_array(getattr(result, "best_phases"))


def _result_score(result: Any) -> float:
    return float(getattr(result, "best_score"))


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
    cell = np.empty((1, 7), dtype=object)
    cell[0, 0] = np.concatenate(
        [[free_precession_t180], segment_lengths, [free_precession_t180]]
    )
    cell[0, 1] = np.concatenate([[0.0], phases, [0.0]])
    cell[0, 2] = np.concatenate([[0.0], np.ones(phases.size), [0.0]])
    cell[0, 3] = _result_score(result)
    cell[0, 4] = _summary_struct(multistart, result, index)
    cell[0, 5] = {
        "initial_phases": _as_array(getattr(result, "initial_phases", phases))
    }
    cell[0, 6] = {
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
    cells, while excitation and inverse-excitation searches produce `1 x 10`
    cells. Times are stored in T180-normalized units unless the caller supplies
    a different convention through the segment/free-precession arguments.
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
