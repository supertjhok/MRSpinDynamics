"""EFG convergence study: is a low predicted eta just under-convergence?

The asymmetry parameter ``eta = (V_xx - V_yy) / V_zz`` is a normalized difference
of the two smaller EFG eigenvalues, so it converges with plane-wave / PAW-grid
cutoff and k-point sampling **much more slowly** than ``C_Q`` (which tracks the
largest eigenvalue).  A static run can reproduce ``C_Q`` while still reporting a
too-axially-symmetric (low) eta simply because the transverse components are not
yet converged.  This driver isolates that effect: it sweeps ``ecut``,
``pawecutdg`` and ``ngkpt`` one at a time around the baseline in a base ABINIT
input, then reports **eta** (and ``C_Q``) versus each knob so the numerical
contribution to the eta gap can be read off before reaching for physics.

See ``docs/eta_accuracy_improvement.md`` (Tier 1, item 1) for the wider plan.

Two stages with real ABINIT runs in between:

  generate -- read a base static EFG ``.abi``, write one variant input per swept
              value (others held at baseline) plus a manifest.
              [then run ABINIT EFG on every staged ``*.abi``]
  collect  -- parse the variant outputs, extract the target-atom eta and C_Q,
              and tabulate them per knob with successive deltas + a converged
              flag.  Writes Markdown + CSV.

Examples
--------
  # from the QuadrupolarDFT root, with PYTHONPATH=src
  python3 examples/abinit/efg_convergence.py generate \\
      --base examples/abinit/nano2_efg.abi --target 2 \\
      --pawecutdg 50,60,80,100 --ngkpt 4,6,8 --ecut 25,30,35 \\
      --out runs/nano2_conv

  bash examples/abinit/run_convergence_efg_wsl.sh runs/nano2_conv

  python3 examples/abinit/efg_convergence.py collect \\
      --workdir runs/nano2_conv --quadmom 0.02044 \\
      --out runs/nano2_conv/convergence.md --csv runs/nano2_conv/convergence.csv

``--target`` is the 0-based atom index (ABINIT atom indices are 1-based); for the
starter NaNO2 cell the first nitrogen is index 2.  No synthetic data: every eta
and C_Q comes from your ABINIT outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from quadrupolar_dft import (
    coupling_constant_hz,
    efg_tensor_from_record,
    parse_abinit_efg,
)

# Default: consider eta converged once successive steps move it by less than this.
DEFAULT_ETA_TOL = 0.01


# ---------------------------------------------------------------------------
# ABINIT input rewriting
# ---------------------------------------------------------------------------
def _scalar_regex(name: str) -> re.Pattern[str]:
    # Anchor at line start so `ecut` never matches inside `pawecutdg`, and skip
    # comment lines (they start with `#`). Keep any trailing inline comment.
    return re.compile(
        rf"(?im)^([ \t]*){re.escape(name)}\b([ \t]+)(\S+)(.*)$"
    )


def set_scalar(text: str, name: str, value) -> str:
    """Set an ABINIT scalar variable (first occurrence), appending if absent."""

    new_text, n = _scalar_regex(name).subn(
        lambda m: f"{m.group(1)}{name}{m.group(2)}{value}{m.group(4)}", text, count=1
    )
    if n == 0:
        new_text = text.rstrip("\n") + f"\n{name} {value}\n"
    return new_text


def get_scalar(text: str, name: str) -> float | None:
    """Read an ABINIT scalar variable's numeric value, or None if absent."""

    match = _scalar_regex(name).search(text)
    if match is None:
        return None
    try:
        return float(match.group(3).replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


_NGKPT_RE = re.compile(
    r"(?im)^([ \t]*)ngkpt\b([ \t]+)(\S+)[ \t]+(\S+)[ \t]+(\S+)(.*)$"
)


def set_ngkpt(text: str, ka: int, kb: int, kc: int) -> str:
    """Set the three ``ngkpt`` components (first occurrence), appending if absent."""

    new_text, n = _NGKPT_RE.subn(
        lambda m: f"{m.group(1)}ngkpt{m.group(2)}{ka} {kb} {kc}{m.group(6)}",
        text,
        count=1,
    )
    if n == 0:
        new_text = text.rstrip("\n") + f"\nngkpt {ka} {kb} {kc}\n"
    return new_text


def get_ngkpt(text: str) -> list[int] | None:
    match = _NGKPT_RE.search(text)
    if match is None:
        return None
    return [int(float(match.group(i))) for i in (3, 4, 5)]


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------
def _parse_ngkpt_token(token: str) -> tuple[int, int, int]:
    """Parse a k-mesh token: ``6`` -> (6,6,6); ``4x6x6`` -> (4,6,6)."""

    parts = token.lower().split("x")
    if len(parts) == 1:
        k = int(parts[0])
        return (k, k, k)
    if len(parts) == 3:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"bad ngkpt token {token!r}: use N or AxBxC")


def _tag(value) -> str:
    """Filename-safe tag for a numeric value (25 -> '25', 2.5 -> '2p5')."""

    if isinstance(value, tuple):
        return "x".join(str(v) for v in value)
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _fmt_num(value: float) -> str:
    return f"{value:g}"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def cmd_generate(args) -> None:
    base_path = Path(args.base)
    base = base_path.read_text(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_ecut = get_scalar(base, "ecut")
    base_pawecutdg = get_scalar(base, "pawecutdg")
    base_ngkpt = get_ngkpt(base)

    ecut_values = _floats(args.ecut)
    pawecutdg_values = _floats(args.pawecutdg)
    ngkpt_values = [_parse_ngkpt_token(t) for t in args.ngkpt.split(",")] if args.ngkpt else []

    # Anchor the baseline from the base input, falling back to the first swept
    # value of a knob when the base does not set it.
    if base_ecut is None:
        base_ecut = ecut_values[0] if ecut_values else None
    if base_pawecutdg is None:
        base_pawecutdg = pawecutdg_values[0] if pawecutdg_values else None
    if base_ngkpt is None and ngkpt_values:
        base_ngkpt = list(ngkpt_values[0])
    if base_ecut is None or base_pawecutdg is None or base_ngkpt is None:
        raise SystemExit(
            "base input must define ecut, pawecutdg and ngkpt (or supply a sweep "
            "value for the missing knob so the baseline can be anchored)."
        )
    baseline = {
        "ecut": base_ecut,
        "pawecutdg": base_pawecutdg,
        "ngkpt": [int(k) for k in base_ngkpt],
    }

    jobs: list[dict] = []

    def _write_job(name: str, swept: str, ecut, pawecutdg, ngkpt) -> None:
        text = set_scalar(base, "ecut", _fmt_num(ecut))
        text = set_scalar(text, "pawecutdg", _fmt_num(pawecutdg))
        text = set_ngkpt(text, *ngkpt)
        (out / f"{name}.abi").write_text(text, encoding="utf-8")
        jobs.append(
            {
                "name": name,
                "swept": swept,
                "ecut": float(ecut),
                "pawecutdg": float(pawecutdg),
                "ngkpt": [int(k) for k in ngkpt],
            }
        )

    # One baseline job, shared as the anchor of every knob's ladder.
    _write_job("baseline", "baseline", baseline["ecut"], baseline["pawecutdg"], baseline["ngkpt"])

    for value in ecut_values:
        if _close(value, baseline["ecut"]):
            continue
        _write_job(f"ecut_{_tag(value)}", "ecut", value, baseline["pawecutdg"], baseline["ngkpt"])
    for value in pawecutdg_values:
        if _close(value, baseline["pawecutdg"]):
            continue
        _write_job(f"pawecutdg_{_tag(value)}", "pawecutdg", baseline["ecut"], value, baseline["ngkpt"])
    for triple in ngkpt_values:
        if list(triple) == baseline["ngkpt"]:
            continue
        _write_job(f"ngkpt_{_tag(triple)}", "ngkpt", baseline["ecut"], baseline["pawecutdg"], triple)

    manifest = {
        "base": str(base_path),
        "target_atom_index": int(args.target),
        "baseline": baseline,
        "jobs": jobs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(jobs)} convergence input(s) + manifest.json to {out}")
    print(f"  baseline: ecut={baseline['ecut']:g} pawecutdg={baseline['pawecutdg']:g} "
          f"ngkpt={baseline['ngkpt']}")
    print(f"  target atom index {args.target} (0-based; ABINIT atom {args.target + 1})")
    print(f"Next: bash examples/abinit/run_convergence_efg_wsl.sh {out}")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def _record_for_atom(text: str, abinit_atom: int):
    records = parse_abinit_efg(text)
    return next((r for r in records if r.atom_index == abinit_atom), None)


def cmd_collect(args) -> None:
    workdir = Path(args.workdir)
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    target = int(manifest["target_atom_index"])
    abinit_atom = target + 1

    rows: list[dict] = []
    for job in manifest["jobs"]:
        name = job["name"]
        path = workdir / f"{name}{args.suffix}"
        if not path.exists():
            if args.allow_missing:
                print(f"  (skipping missing output {path.name})")
                continue
            raise FileNotFoundError(
                f"missing ABINIT output for {name!r}: {path} "
                f"(run run_convergence_efg_wsl.sh, or pass --allow-missing)"
            )
        record = _record_for_atom(path.read_text(encoding="utf-8"), abinit_atom)
        if record is None:
            raise ValueError(f"no EFG record for atom {abinit_atom} in {path.name}")
        tensor = efg_tensor_from_record(record)
        cq_hz = coupling_constant_hz(tensor.vzz_si, args.quadmom)
        rows.append(
            {
                **{k: job[k] for k in ("name", "swept", "ecut", "pawecutdg", "ngkpt")},
                "eta": float(tensor.eta),
                "cq_mhz": cq_hz / 1e6,
                "eta_abinit": float(record.eta),
                "cq_abinit_mhz": float(record.cq_mhz),
            }
        )

    if not rows:
        raise SystemExit("no outputs collected; nothing to report.")

    baseline_row = next((r for r in rows if r["swept"] == "baseline"), None)
    report = _build_report(manifest, rows, baseline_row, args.eta_tol)
    print(report)

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}")
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"Wrote CSV to {args.csv}")


def _knob_value(row: dict, knob: str):
    if knob == "ngkpt":
        return tuple(row["ngkpt"])
    return row[knob]


def _build_report(manifest: dict, rows: list[dict], baseline_row, eta_tol: float) -> str:
    out: list[str] = []

    def w(text: str = "") -> None:
        out.append(text)

    baseline = manifest["baseline"]
    w("# EFG convergence study: eta and C_Q vs numerical knobs")
    w()
    w(f"- Base input: `{manifest['base']}`")
    w(f"- Target atom index {manifest['target_atom_index']} (0-based; "
      f"ABINIT atom {manifest['target_atom_index'] + 1}).")
    w(f"- Baseline: ecut={baseline['ecut']:g} Ha, "
      f"pawecutdg={baseline['pawecutdg']:g} Ha, ngkpt={baseline['ngkpt']}.")
    if baseline_row is not None:
        w(f"- Baseline result: **eta = {baseline_row['eta']:.4f}**, "
          f"C_Q = {baseline_row['cq_mhz']:+.4f} MHz.")
    w()
    w("eta is a difference of the two smaller EFG eigenvalues, so it converges "
      "more slowly than C_Q. A drift in eta across a ladder means the transverse "
      "components are not yet converged; a flat tail means they are.")
    w()

    for knob, unit in (("ecut", "Ha"), ("pawecutdg", "Ha"), ("ngkpt", "")):
        ladder = [r for r in rows if r["swept"] in (knob, "baseline")]
        # Keep one row per distinct knob value (baseline may coincide with a sweep).
        seen: dict = {}
        for r in ladder:
            seen[_knob_value(r, knob)] = r
        ladder = sorted(seen.values(), key=lambda r: _knob_value(r, knob))
        if len(ladder) < 2:
            continue

        w(f"## Sweep: {knob}")
        w()
        header_val = "ngkpt" if knob == "ngkpt" else f"{knob} ({unit})"
        w(f"| {header_val} | eta | d(eta) | C_Q (MHz) | d(C_Q) |")
        w("|---|---|---|---|---|")
        prev = None
        converged_at = None
        for r in ladder:
            val = _knob_value(r, knob)
            val_str = "x".join(str(k) for k in val) if knob == "ngkpt" else f"{val:g}"
            if prev is None:
                d_eta_str, d_cq_str = "--", "--"
            else:
                d_eta = r["eta"] - prev["eta"]
                d_cq = r["cq_mhz"] - prev["cq_mhz"]
                d_eta_str = f"{d_eta:+.4f}"
                d_cq_str = f"{d_cq:+.4f}"
                if abs(d_eta) < eta_tol and converged_at is None:
                    converged_at = val
            w(f"| {val_str} | {r['eta']:.4f} | {d_eta_str} | "
              f"{r['cq_mhz']:+.4f} | {d_cq_str} |")
            prev = r
        w()
        first, last = ladder[0], ladder[-1]
        total_shift = last["eta"] - first["eta"]
        if converged_at is not None:
            conv_str = ("x".join(str(k) for k in converged_at)
                        if knob == "ngkpt" else f"{converged_at:g}")
            w(f"- eta shifts {total_shift:+.4f} across the ladder; successive "
              f"change falls below {eta_tol:g} by {knob} = {conv_str}.")
        else:
            w(f"- eta shifts {total_shift:+.4f} across the ladder and has **not** "
              f"settled below {eta_tol:g} -- extend the {knob} sweep further.")
        w()

    w("## Reading the result")
    w()
    w("- If eta keeps climbing as pawecutdg / ngkpt increase, part of the "
      "measured-vs-DFT eta gap was numerical: take the converged tail value as "
      "the honest static prediction before comparing to experiment or invoking "
      "Tier 2+ physics (dispersion, geometry, dynamics).")
    w("- If eta is already flat at the baseline, the gap is physical, not "
      "numerical -- see `docs/eta_accuracy_improvement.md`, Tier 2 onward.")
    w()
    return "\n".join(out)


def _write_csv(path: str, rows: list[dict]) -> None:
    fields = ["name", "swept", "ecut", "pawecutdg", "ngkpt",
              "eta", "cq_mhz", "eta_abinit", "cq_abinit_mhz"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["ngkpt"] = "x".join(str(k) for k in row["ngkpt"])
            writer.writerow(record)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _floats(spec: str | None) -> list[float]:
    if not spec:
        return []
    return [float(x) for x in spec.split(",") if x.strip()]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="stage variant EFG inputs sweeping cutoffs / k-mesh")
    g.add_argument("--base", required=True, help="base static EFG .abi input")
    g.add_argument("--target", type=int, required=True, help="0-based target atom index")
    g.add_argument("--ecut", help="comma list of ecut values (Ha), e.g. 25,30,35")
    g.add_argument("--pawecutdg", help="comma list of pawecutdg values (Ha), e.g. 50,60,80,100")
    g.add_argument("--ngkpt", help="comma list of k-meshes: N or AxBxC, e.g. 4,6,8 or 4x4x4,6x6x6")
    g.add_argument("--out", required=True, help="workdir for the staged inputs")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("collect", help="parse variant outputs -> eta/C_Q convergence tables")
    c.add_argument("--workdir", required=True)
    c.add_argument("--quadmom", type=float, default=0.02044, help="barns (default 14N)")
    c.add_argument("--suffix", default=".abo")
    c.add_argument("--eta-tol", type=float, default=DEFAULT_ETA_TOL,
                   help=f"eta step below which a knob is called converged (default {DEFAULT_ETA_TOL})")
    c.add_argument("--allow-missing", action="store_true",
                   help="skip variants whose ABINIT output is missing")
    c.add_argument("--out", help="write the Markdown report to this path")
    c.add_argument("--csv", help="also write the per-variant rows to this CSV")
    c.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
