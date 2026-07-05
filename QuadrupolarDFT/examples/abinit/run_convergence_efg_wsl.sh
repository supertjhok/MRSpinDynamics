#!/usr/bin/env bash
# Run ABINIT EFG on every *.abi in a staged convergence-study directory.
#
# Usage:
#   bash run_convergence_efg_wsl.sh runs/nano2_conv
#   bash run_convergence_efg_wsl.sh runs/nano2_conv --dry-run
#
# The repo workdir is typically on a OneDrive-synced /mnt/c (DrvFs) mount, where
# ABINIT's scratch and .abo I/O is very slow and OneDrive file locks can stall a
# run (no .abo ever appears). To avoid that, each job runs in a WSL-native scratch
# directory (off /mnt/c) and only the EFG output + logs are copied back into the
# repo workdir -- which is where `efg_convergence.py collect` looks for
# <name>.abo.
#
# Environment overrides:
#   ABINIT_SCRATCH_DIR   scratch root to use (default: a mktemp dir under
#                        ${TMPDIR:-/tmp}). Must be a WSL-native path, not /mnt/c.
#   ABINIT_KEEP_SCRATCH  1 (default) keeps the scratch after a successful run;
#                        set 0 to auto-remove it (only when it was auto-created).
#   ABINIT_NP            serial (unset/1), a fixed count, or `auto` (see
#                        abinit_parallel.sh).
#
# The variant inputs differ in ecut / pawecutdg / ngkpt, so -- unlike the
# finite-displacement runner -- the MPI launch command is resolved per input
# (the IBZ k-point count changes with ngkpt).
set -uo pipefail

# Shared MPI-launch helpers (resolve the path before any cd).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/abinit_parallel.sh"

workdir="${1:-}"
if [[ -z "$workdir" ]]; then
  echo "usage: run_convergence_efg_wsl.sh <workdir> [--dry-run]" >&2
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

# WSL-native scratch root, off any /mnt/c DrvFs / OneDrive path.
scratch_root="${ABINIT_SCRATCH_DIR:-}"
auto_scratch=0
if [[ -z "$scratch_root" ]]; then
  scratch_root="$(mktemp -d "${TMPDIR:-/tmp}/qdft_conv.XXXXXX")" \
    || { echo "ERROR: could not create scratch dir under ${TMPDIR:-/tmp}" >&2; exit 1; }
  auto_scratch=1
else
  mkdir -p "$scratch_root" \
    || { echo "ERROR: cannot create ABINIT_SCRATCH_DIR=$scratch_root" >&2; exit 1; }
fi
case "$scratch_root" in
  /mnt/*)
    echo "WARNING: scratch $scratch_root is on a Windows mount; expect slow I/O." >&2
    echo "         Set ABINIT_SCRATCH_DIR to a WSL-native path (e.g. \$HOME/qdft_scratch)." >&2
    ;;
esac
case "$workdir" in
  /mnt/*) echo "Note: workdir $workdir is on a Windows mount; running in local scratch and copying results back." >&2 ;;
esac
echo "Scratch: $scratch_root"
echo "Running ABINIT on ${#inputs[@]} input(s); results -> $workdir"

done_count=0
failed=0
for input in "${inputs[@]}"; do
  name="$(basename "$input" .abi)"
  job_dir="$scratch_root/$name"
  mkdir -p "$job_dir"
  cp "$input" "$job_dir/$name.abi"
  echo "==> $name"
  # ngkpt varies between variants, so re-probe the launch command each time.
  abinit_build_cmd "$job_dir/$name.abi" || { failed=1; break; }
  extra=()
  (( dry_run )) && extra=(--dry-run)
  run_ok=1
  ( cd "$job_dir" && "${abinit_cmd[@]}" "$name.abi" "${extra[@]}" \
      > "$name.stdout" 2> "$name.stderr" ) || run_ok=0

  # Always copy logs back so a run can be inspected from the repo side.
  result_dir="$workdir/$name.run"
  mkdir -p "$result_dir"
  cp -f "$job_dir/$name.stdout" "$job_dir/$name.stderr" "$result_dir/" 2>/dev/null || true

  if (( ! run_ok )); then
    echo "ERROR: ABINIT failed on $name. Diagnostic from $name.stdout:" >&2
    grep -A3 -iE "ERROR|Action:" "$job_dir/$name.stdout" 2>/dev/null | head -20 >&2
    echo "(Logs copied to $result_dir; scratch kept at $job_dir. Fix and re-run.)" >&2
    failed=1
    break
  fi

  if [[ -f "$job_dir/$name.abo" ]]; then
    # <name>.abo flat in the workdir is where `collect` looks; keep a copy in the
    # per-job folder too for provenance.
    cp -f "$job_dir/$name.abo" "$workdir/$name.abo"
    cp -f "$job_dir/$name.abo" "$result_dir/$name.abo" 2>/dev/null || true
  elif (( ! dry_run )); then
    echo "WARNING: no $name.abo produced in $job_dir (see $result_dir/$name.stdout)." >&2
  fi
  done_count=$((done_count + 1))
done

echo "Completed $done_count/${#inputs[@]} runs."
if (( failed )); then
  echo "Stopped early after a failure; scratch kept at $scratch_root" >&2
  exit 1
fi

if [[ "${ABINIT_KEEP_SCRATCH:-1}" == "0" && "$auto_scratch" == "1" ]]; then
  rm -rf "$scratch_root"
  echo "Removed scratch $scratch_root (ABINIT_KEEP_SCRATCH=0)."
else
  echo "Scratch retained at $scratch_root"
fi
(( dry_run )) || echo "Now run: python3 examples/abinit/efg_convergence.py collect --workdir $workdir"
