#!/usr/bin/env bash
# Run one ABINIT static EFG input and copy the .abo beside the .abi.
#
# Usage:
#   bash examples/abinit/run_static_efg_wsl.sh runs/glycine_static/glycine_efg.abi
#   bash examples/abinit/run_static_efg_wsl.sh \
#     runs/glycine_static/glycine_efg.abi --dry-run
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/abinit_parallel.sh"

input="${1:-}"
if [[ -z "$input" ]]; then
  echo "usage: run_static_efg_wsl.sh <input.abi> [--dry-run]" >&2
  exit 2
fi
dry_run=0
[[ "${2:-}" == "--dry-run" ]] && dry_run=1

if [[ ! -f "$input" ]]; then
  echo "ERROR: input not found: $input" >&2
  exit 1
fi
if ! command -v abinit >/dev/null 2>&1; then
  echo "ERROR: 'abinit' not on PATH (run inside WSL, not Git Bash)." >&2
  exit 1
fi

input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
workdir="$(dirname "$input")"
name="$(basename "$input" .abi)"
job_dir="$workdir/$name.run"
export ABI_PSPDIR="${ABI_PSPDIR:-/usr/share/abinit/psp}"

abinit_build_cmd "$input" || exit 1

mkdir -p "$job_dir"
cp "$input" "$job_dir/$name.abi"
extra=()
(( dry_run )) && extra=(--dry-run)

echo "Running ABINIT static EFG input: $input"
if ! (
  cd "$job_dir" \
    && "${abinit_cmd[@]}" "$name.abi" "${extra[@]}" > "$name.stdout" 2> "$name.stderr"
); then
  echo "ERROR: ABINIT failed. Diagnostic from $name.stdout:" >&2
  grep -A3 -iE "ERROR|Action:" "$job_dir/$name.stdout" 2>/dev/null | head -20 >&2
  exit 1
fi

if [[ -f "$job_dir/$name.abo" ]]; then
  cp "$job_dir/$name.abo" "$workdir/$name.abo"
fi

if (( dry_run )); then
  echo "ABINIT dry-run completed for $name."
else
  echo "Completed $name. Output: $workdir/$name.abo"
fi
