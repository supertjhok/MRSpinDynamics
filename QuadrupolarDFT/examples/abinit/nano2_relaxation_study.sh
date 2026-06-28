#!/usr/bin/env bash
# Relaxed-vs-unrelaxed NaNO2 14N finite-temperature EFG study (real ABINIT runs).
#
# Goal: a CLEAN comparison that changes ONLY the geometry. By default the study is
# self-contained -- it runs the full finite-displacement workflow on BOTH the
# unrelaxed starter geometry and the relaxed geometry, with identical settings
# (same target atom, modes, displacement amplitude, temperatures, Q, and DFT
# inputs), so only the structure differs. The objective is to see how much
# relaxing to an energy minimum improves accuracy (eta -> ~0.38, the two strong
# lines splitting toward 3.60 / 4.64 MHz).
#
# If you ALREADY have a matching unrelaxed run, pass --reuse-unrelaxed <dir> to
# reuse it instead of recomputing: its EFG outputs are re-collected (no ABINIT)
# and its displacement parameters (target / #modes / amplitude) are read from its
# manifest and applied to the relaxed branch, so the two sides stay identical.
#
# ABINIT is parallelized automatically: by default it picks the best `mpirun -np`
# from the core count and the run's k-points (falling back to serial if no MPI is
# found). Pin it with --np N, or force serial with --np 1.
#
# Run inside WSL (abinit + anaddb must be on PATH), from anywhere:
#   bash examples/abinit/nano2_relaxation_study.sh                  # compute both (auto -np)
#   bash examples/abinit/nano2_relaxation_study.sh --np 18          # pin 18 MPI processes
#   bash examples/abinit/nano2_relaxation_study.sh --np 1           # force serial
#   bash examples/abinit/nano2_relaxation_study.sh --dry-run        # validate inputs
#   bash examples/abinit/nano2_relaxation_study.sh --study-dir runs/my_study
#   bash examples/abinit/nano2_relaxation_study.sh --reuse-unrelaxed runs/nano2_disp
#
# Outputs (under <study-dir>, default runs/nano2_relax_study):
#   unrelaxed.json, relaxed.json         per-branch results
#   relaxation_comparison.md             the comparison report
#   unrelaxed/, relaxed/                 the staged inputs and ABINIT outputs
#
# This is a long calculation: a DFPT phonon run plus ~2*max_modes+1 EFG runs per
# branch (the unrelaxed branch is skipped only when --reuse-unrelaxed is given).
set -euo pipefail

# ---- resolve the QuadrupolarDFT root (two levels up from this script) --------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root/src:${PYTHONPATH:-}"

# ---- defaults / arguments ----------------------------------------------------
base="examples/abinit/nano2_efg.abi"
study_dir="runs/nano2_relax_study"
target=2                       # 0-based index of the resonant 14N atom
quadmom=0.02044                # 14N quadrupole moment (barns)
spin=1.0                       # 14N
max_modes=6
max_disp=0.04
temperatures="0,77,150,300"
dry_run=0
reuse_unrelaxed=""             # if set (--reuse-unrelaxed DIR), reuse that run

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) base="$2"; shift 2 ;;
    --study-dir) study_dir="$2"; shift 2 ;;
    --target) target="$2"; shift 2 ;;
    --quadmom) quadmom="$2"; shift 2 ;;
    --spin) spin="$2"; shift 2 ;;
    --max-modes) max_modes="$2"; shift 2 ;;
    --max-displacement) max_disp="$2"; shift 2 ;;
    --temperatures) temperatures="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --np) export ABINIT_NP="$2"; shift 2 ;;   # MPI processes for the ABINIT runs
    --reuse-unrelaxed) reuse_unrelaxed="$2"; shift 2 ;;
    -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Parallelize by default: ABINIT_NP=auto picks the best -np from the core count
# and the run's k-points, and falls back to serial if no MPI is available. Pin a
# value with --np N, or force serial with --np 1.
: "${ABINIT_NP:=auto}"
export ABINIT_NP

cli="examples/abinit/efg_temperature.py"
run_relax="examples/abinit/run_relax_wsl.sh"
run_phonon="examples/abinit/run_phonon_wsl.sh"
run_fd="examples/abinit/run_finite_displacement_wsl.sh"

if ! command -v abinit >/dev/null 2>&1; then
  echo "ERROR: 'abinit'/'anaddb' not on PATH. Run this inside WSL, not Git Bash." >&2
  exit 1
fi
if [[ ! -f "$base" ]]; then
  echo "ERROR: base input not found: $base" >&2
  exit 1
fi
mkdir -p "$study_dir"

step() { echo; echo "==================== $* ===================="; }

# A full chain (phonon -> displace -> collect) on one base input. Args:
#   $1 = base .abi   $2 = branch dir   $3 = label   $4 = output JSON path
run_chain() {
  local chain_base="$1" dir="$2" label="$3" out_json="$4"
  local ph="$dir/phonon" disp="$dir/disp"

  step "$label: phonon (DFPT + anaddb)"
  python3 "$cli" phonon --base "$chain_base" --out "$ph"
  bash "$run_phonon" "$ph"

  step "$label: displaced EFG inputs"
  python3 "$cli" displace --base "$chain_base" --anaddb "$ph/anaddb.out" \
      --target "$target" --max-modes "$max_modes" --max-displacement "$max_disp" \
      --out "$disp"
  bash "$run_fd" "$disp"

  step "$label: collect -> $out_json"
  python3 "$cli" collect --workdir "$disp" --temperatures "$temperatures" \
      --spin "$spin" --quadmom "$quadmom" --label "$label" --out-json "$out_json"
}

# ---- dry-run: stage and validate the relaxed-branch inputs -------------------
if (( dry_run )); then
  step "DRY RUN: validating ABINIT inputs (no full SCF)"
  export ABINIT_NP=1   # input validation is cheap; don't spin up MPI for it
  if [[ -n "$reuse_unrelaxed" && ! -f "$reuse_unrelaxed/manifest.json" ]]; then
    echo "WARNING: --reuse-unrelaxed '$reuse_unrelaxed' has no manifest.json." >&2
  fi
  python3 "$cli" relax --base "$base" --out "$study_dir/relaxed/relax"
  bash "$run_relax" "$study_dir/relaxed/relax" --dry-run
  python3 "$cli" phonon --base "$base" --out "$study_dir/relaxed/phonon"
  bash "$run_phonon" "$study_dir/relaxed/phonon" --dry-run
  echo
  echo "Dry run OK: relaxation and phonon inputs parse and pass abinit --dry-run."
  echo "Re-run without --dry-run for the full study (this takes a while)."
  exit 0
fi

# ---- unrelaxed side ----------------------------------------------------------
# Default: compute the unrelaxed branch from scratch with the same settings as the
# relaxed branch (run_chain uses the shared $target/$max_modes/$max_disp), so the
# two sides are identical except for the geometry. If --reuse-unrelaxed DIR is
# given, reuse that existing run instead: re-collect its EFG outputs into JSON (no
# DFT) and lock the displacement parameters from its manifest onto the relaxed
# branch, so the comparison stays clean against a run you already have.
if [[ -z "$reuse_unrelaxed" ]]; then
  run_chain "$base" "$study_dir/unrelaxed" "unrelaxed" "$study_dir/unrelaxed.json"
else
  step "unrelaxed: reuse existing run ($reuse_unrelaxed)"
  if [[ ! -f "$reuse_unrelaxed/manifest.json" ]]; then
    echo "ERROR: no manifest.json under '$reuse_unrelaxed'." >&2
    echo "  Point --reuse-unrelaxed at the earlier displaced workdir, or drop the" >&2
    echo "  flag to compute the unrelaxed branch from scratch." >&2
    exit 1
  fi
  # Re-collect the existing outputs (cheap; no ABINIT). Reuses temps/Q/spin so
  # both sides are post-processed identically.
  python3 "$cli" collect --workdir "$reuse_unrelaxed" --temperatures "$temperatures" \
      --spin "$spin" --quadmom "$quadmom" --label "unrelaxed" \
      --out-json "$study_dir/unrelaxed.json"
  # Lock the displacement parameters to the reused run's manifest.
  read -r target max_modes max_disp < <(
    python3 - "$reuse_unrelaxed/manifest.json" "$base" <<'PY'
import json, sys
import numpy as np
from quadrupolar_dft import PhononMode, parse_abinit_structure
from quadrupolar_dft.finite_displacement import cartesian_step_angstrom
manifest = json.load(open(sys.argv[1]))
masses = parse_abinit_structure(open(sys.argv[2]).read()).masses_amu
dq = {j["mode_index"]: j["delta_q_si"]
      for j in manifest["jobs"] if j["mode_index"] >= 0 and j["sign"] == 1}
disps = []
for i, mode in enumerate(manifest["modes"]):
    pm = PhononMode(mode["wavenumber_cm_inv"], np.array(mode["eigenvector"]))
    disps.append(float(np.max(np.linalg.norm(cartesian_step_angstrom(pm, masses, dq[i]), axis=1))))
print(int(manifest["target_atom_index"]), len(manifest["modes"]), round(max(disps), 6))
PY
  )
  echo "  locked from reused run: target=$target  max_modes=$max_modes  max_displacement=$max_disp A"
  echo "  (relaxed branch will use these identical settings; only the geometry differs.)"
fi

# ---- relaxed branch ----------------------------------------------------------
step "relaxed: structure relaxation (Stage 0)"
relax_dir="$study_dir/relaxed/relax"
python3 "$cli" relax --base "$base" --out "$relax_dir"
bash "$run_relax" "$relax_dir"
python3 "$cli" relax-collect --base "$base" --abo "$relax_dir/relax.abo" --out "$relax_dir"
relaxed_abi="$relax_dir/relaxed.abi"

run_chain "$relaxed_abi" "$study_dir/relaxed" "relaxed" "$study_dir/relaxed.json"

# ---- comparison --------------------------------------------------------------
if [[ -f "$study_dir/unrelaxed.json" && -f "$study_dir/relaxed.json" ]]; then
  step "comparison report"
  python3 examples/abinit/compare_relaxation.py \
      "$study_dir/unrelaxed.json" "$study_dir/relaxed.json" \
      --out "$study_dir/relaxation_comparison.md"
  echo
  echo "Study complete. See $study_dir/relaxation_comparison.md"
else
  echo
  echo "Relaxed branch complete: $study_dir/relaxed.json"
  echo "(Unrelaxed JSON missing; if you used --reuse-unrelaxed, check that path.)"
fi
