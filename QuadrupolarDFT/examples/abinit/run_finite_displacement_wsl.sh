#!/usr/bin/env bash
# Run ABINIT EFG on every *.abi in a staged finite-displacement directory.
#
# Usage:
#   bash run_finite_displacement_wsl.sh runs/nano2_disp
#   bash run_finite_displacement_wsl.sh runs/nano2_disp --dry-run
#
# Each input <name>.abi is run in its own subdirectory so ABINIT's scratch and
# output files do not collide; the EFG output is copied back to <name>.abo next
# to the input, which is where `efg_temperature.py collect` looks for it.
set -uo pipefail

workdir="${1:-}"
if [[ -z "$workdir" ]]; then
  echo "usage: run_finite_displacement_wsl.sh <workdir> [--dry-run]" >&2
  exit 2
fi
dry_run=0
[[ "${2:-}" == "--dry-run" ]] && dry_run=1

if [[ ! -d "$workdir" ]]; then
  echo "ERROR: directory not found: $workdir (run from the QuadrupolarDFT root)" >&2
  exit 1
fi
workdir="$(cd "$workdir" && pwd)"
if ! command -v abinit >/dev/null 2>&1; then
  echo "ERROR: 'abinit' not on PATH (run inside WSL, not Git Bash)." >&2
  exit 1
fi
export ABI_PSPDIR="${ABI_PSPDIR:-/usr/share/abinit/psp}"

shopt -s nullglob
inputs=("$workdir"/*.abi)
if (( ${#inputs[@]} == 0 )); then
  echo "No .abi inputs in $workdir" >&2
  exit 1
fi

echo "Running ABINIT on ${#inputs[@]} input(s) in $workdir"
done_count=0
for input in "${inputs[@]}"; do
  name="$(basename "$input" .abi)"
  job_dir="$workdir/$name.run"
  mkdir -p "$job_dir"
  cp "$input" "$job_dir/$name.abi"
  echo "==> $name"
  extra=()
  (( dry_run )) && extra=(--dry-run)
  if ! ( cd "$job_dir" && abinit "${extra[@]}" "$name.abi" > "$name.stdout" 2> "$name.stderr" ); then
    echo "ERROR: ABINIT failed on $name. Diagnostic from $name.stdout:" >&2
    grep -A3 -iE "ERROR|Action:" "$job_dir/$name.stdout" 2>/dev/null | head -20 >&2
    echo "(Fix the input and re-run; outputs already produced are kept.)" >&2
    exit 1
  fi
  if [[ -f "$job_dir/$name.abo" ]]; then
    cp "$job_dir/$name.abo" "$workdir/$name.abo"
  fi
  done_count=$((done_count + 1))
done
echo "Completed $done_count/${#inputs[@]} runs."
(( dry_run )) || echo "Now run: python3 efg_temperature.py collect --workdir $workdir"
