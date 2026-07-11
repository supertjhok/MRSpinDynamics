"""Quickstart for the unified experiment facade.

This is the recommended on-ramp: describe an experiment with declarative
specs, inspect the plan (resolved workflow, compatibility warnings, and a
runtime/memory estimate) *before* running, execute it, and save the result
with provenance. The facade delegates to the same validated ``run_*``
workflows, so the numbers match a direct call exactly -- it only adds the
plan/run/save scaffolding around them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.experiment import (
    Acquisition,
    CPMGTrain,
    Experiment,
    Hardware,
    Sample,
    load_run,
)


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=501, help="Offset grid points.")
    parser.add_argument("--num-echoes", type=int, default=8, help="Number of echoes.")
    parser.add_argument("--probe", default="tuned", help="ideal/tuned/untuned/matched.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional .npz path.")
    args = parser.parse_args()

    # 1. Describe the experiment. Every spec field left at its default defers
    #    to the wrapped workflow's own default, so this reproduces
    #    run_<probe>_cpmg_train(...) exactly.
    study = Experiment(
        sequence=CPMGTrain(num_echoes=args.num_echoes),
        sample=Sample(t1_seconds=2.0, t2_seconds=2.0),
        hardware=Hardware(probe=args.probe),
        acquisition=Acquisition(numpts=args.numpts, maxoffs=10.0),
    )

    # 2. Plan first: resolve the workflow, front-load compatibility checks
    #    (e.g. is the offset grid fine enough to avoid rephasing?), and get a
    #    host-calibrated runtime/memory estimate.
    plan = study.plan()
    print("== plan ==")
    print(plan.report())
    if not plan.ok:
        raise SystemExit("plan has errors; not running")

    # 3. Run. Plan warnings are re-emitted; the result wraps the native
    #    workflow result plus provenance (versions, platform, elapsed time).
    record = study.run()
    result = record.result
    peak = np.max(np.abs(result.echo), axis=1)
    print("\n== result ==")
    print(f"workflow: {record.provenance['workflow']}")
    print(f"elapsed: {record.provenance['elapsed_seconds'] * 1e3:.1f} ms")
    print(f"num echoes: {result.mrx.shape[0]}")
    print(f"peak echo magnitudes: {np.array2string(peak, precision=6, separator=', ')}")

    # 4. Save arrays + JSON provenance, then round-trip the spec back.
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        record.save(str(args.save_npz))
        loaded = load_run(str(args.save_npz))
        print(f"\nsaved: {args.save_npz}")
        print(f"reloaded spec matches: {loaded.experiment == study}")


if __name__ == "__main__":
    main()
