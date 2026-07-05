"""Command-line runner for config-driven experiments.

Run with ``python -m spin_dynamics.experiment <command>``:

    plan   CONFIG            resolve + validate a config, print the plan
    run    CONFIG [-o NPZ]   plan then run, optionally saving the result
    show   RUN.npz           print a saved run's spec and provenance
    convert CONFIG OUT       rewrite a config between .toml and .json

Configs are the human-friendly TOML/JSON form documented in
:mod:`spin_dynamics.experiment.config`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from spin_dynamics.experiment.config import (
    ConfigError,
    dumps_toml,
    experiment_to_config,
    load_config,
    save_config,
)
from spin_dynamics.experiment.io import load_run
from spin_dynamics.experiment.runner import ExperimentPlanError


def _cmd_plan(args: argparse.Namespace) -> int:
    experiment = load_config(args.config)
    plan = experiment.plan()
    print(plan.report())
    return 0 if plan.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    experiment = load_config(args.config)
    plan = experiment.plan()
    print(plan.report())
    if not plan.ok:
        print("\nplan has errors; not running", file=sys.stderr)
        return 1
    try:
        record = experiment.run()
    except ExperimentPlanError as exc:  # pragma: no cover - guarded by plan.ok
        print(str(exc), file=sys.stderr)
        return 1
    result = record.result
    print("\n== result ==")
    print(f"workflow: {record.provenance['workflow']}")
    print(f"elapsed: {record.provenance['elapsed_seconds'] * 1e3:.1f} ms")
    print(f"result type: {type(result).__name__}")
    if args.output is not None:
        record.save(args.output)
        print(f"saved: {args.output}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    loaded = load_run(args.run)
    print("== experiment ==")
    config = experiment_to_config(loaded.experiment)
    print(dumps_toml(config), end="")
    print("== provenance ==")
    for key, value in loaded.provenance.items():
        print(f"{key}: {value}")
    if loaded.unsaved_result_fields:
        print(f"unsaved result fields: {list(loaded.unsaved_result_fields)}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    experiment = load_config(args.config)
    save_config(experiment, args.output)
    print(f"wrote: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spin_dynamics.experiment",
        description="Config-driven runner for the experiment facade.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="resolve and validate a config")
    p_plan.add_argument("config", help="path to a .toml or .json experiment config")
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="plan then run a config")
    p_run.add_argument("config", help="path to a .toml or .json experiment config")
    p_run.add_argument("-o", "--output", help="optional .npz path to save the result")
    p_run.set_defaults(func=_cmd_run)

    p_show = sub.add_parser("show", help="print a saved run's spec and provenance")
    p_show.add_argument("run", help="path to a saved run .npz")
    p_show.set_defaults(func=_cmd_show)

    p_convert = sub.add_parser("convert", help="rewrite a config between .toml/.json")
    p_convert.add_argument("config", help="input .toml or .json config")
    p_convert.add_argument("output", help="output .toml or .json path")
    p_convert.set_defaults(func=_cmd_convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
