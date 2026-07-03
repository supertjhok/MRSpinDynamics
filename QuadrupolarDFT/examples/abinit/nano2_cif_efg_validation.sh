#!/usr/bin/env bash
# Validate NaNO2 CIF ranking with static ABINIT EFG simulations.
#
# The script stages static EFG inputs from the best-ranked NaNO2 CIFs, runs each
# one through the MPI-aware run_static_efg_wsl.sh helper, then collects 14N
# C_Q/eta/line errors into CSV and Markdown reports.
#
# Run inside WSL from anywhere:
#   bash examples/abinit/nano2_cif_efg_validation.sh --dry-run
#   bash examples/abinit/nano2_cif_efg_validation.sh
#   bash examples/abinit/nano2_cif_efg_validation.sh --np 18
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root/src:${PYTHONPATH:-}"

workdir="runs/nano2_cif_efg_validation"
top=5
paraelectric_controls=2
target_temperature=295
ngkpt="4,4,4"
ecut=25
pawecutdg=50
pseudo_dir="Pseudodojo_paw_pbe_standard"
dry_run=0
prepare_only=0
collect_only=0
allow_missing=0
include_args=()
include_partial_occupancy=0
stop_on_failure=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) workdir="$2"; shift 2 ;;
    --top) top="$2"; shift 2 ;;
    --paraelectric-controls) paraelectric_controls="$2"; shift 2 ;;
    --target-temperature) target_temperature="$2"; shift 2 ;;
    --ngkpt) ngkpt="$2"; shift 2 ;;
    --ecut) ecut="$2"; shift 2 ;;
    --pawecutdg) pawecutdg="$2"; shift 2 ;;
    --pseudo-dir) pseudo_dir="$2"; shift 2 ;;
    --include) include_args+=(--include "$2"); shift 2 ;;
    --include-partial-occupancy) include_partial_occupancy=1; shift ;;
    --stop-on-failure) stop_on_failure=1; shift ;;
    --np) export ABINIT_NP="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --prepare-only) prepare_only=1; shift ;;
    --collect-only) collect_only=1; shift ;;
    --allow-missing) allow_missing=1; shift ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${ABINIT_NP:=auto}"
export ABINIT_NP
helper="examples/abinit/nano2_cif_efg_validation.py"
runner="examples/abinit/run_static_efg_wsl.sh"
csv_report="results/nano2_cif_efg_validation.csv"
md_report="results/nano2_cif_efg_validation.md"

if (( ! collect_only )); then
  prepare_extra=()
  (( include_partial_occupancy )) && prepare_extra+=(--include-partial-occupancy)
  python3 "$helper" prepare \
    --workdir "$workdir" \
    --top "$top" \
    --paraelectric-controls "$paraelectric_controls" \
    --target-temperature "$target_temperature" \
    --ngkpt "$ngkpt" \
    --ecut "$ecut" \
    --pawecutdg "$pawecutdg" \
    --pseudo-dir "$pseudo_dir" \
    "${include_args[@]}" \
    "${prepare_extra[@]}"
fi

if (( prepare_only )); then
  echo
  echo "Prepare-only complete. Staged inputs are under $workdir."
  exit 0
fi

if (( ! collect_only )); then
  if ! command -v abinit >/dev/null 2>&1; then
    echo "ERROR: 'abinit' not on PATH. Run this inside WSL, not PowerShell." >&2
    exit 1
  fi
  echo
  echo "ABINIT MPI mode: ABINIT_NP=$ABINIT_NP"
  echo "Override launcher with ABINIT_MPIRUN if needed."
  echo
  mapfile -t inputs < <(python3 "$helper" list-inputs --workdir "$workdir")
  if (( ${#inputs[@]} == 0 )); then
    echo "ERROR: no staged inputs found under $workdir" >&2
    exit 1
  fi
  failed=0
  for input in "${inputs[@]}"; do
    if (( dry_run )); then
      if ! bash "$runner" "$input" --dry-run; then
        failed=1
        (( stop_on_failure )) && exit 1
        echo "WARNING: dry-run failed for $input; continuing." >&2
      fi
    else
      if ! bash "$runner" "$input"; then
        failed=1
        (( stop_on_failure )) && exit 1
        echo "WARNING: ABINIT failed for $input; continuing." >&2
      fi
    fi
  done
  if (( failed )); then
    allow_missing=1
  fi
fi

if (( dry_run )); then
  echo
  echo "Dry run complete. Re-run without --dry-run for full EFG simulations."
  exit 0
fi

collect_args=()
(( allow_missing )) && collect_args+=(--allow-missing)
python3 "$helper" collect \
  --workdir "$workdir" \
  --csv "$csv_report" \
  --markdown "$md_report" \
  "${collect_args[@]}"

echo
echo "Validation complete."
echo "  CSV: $csv_report"
echo "  Markdown: $md_report"
