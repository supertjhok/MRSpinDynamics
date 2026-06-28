#!/usr/bin/env bash
# Run the ABINIT ionic relaxation for a staged relax directory.
#
# Usage (run from the QuadrupolarDFT root, or pass an absolute path):
#   bash examples/abinit/run_relax_wsl.sh runs/nano2_relax
#   bash examples/abinit/run_relax_wsl.sh runs/nano2_relax --dry-run
#
# Produces in <workdir>:
#   relax.abo         ABINIT relaxation output (its footer echoes the relaxed
#                     geometry)  <-- feed this to
#                     efg_temperature.py relax-collect --abo
set -uo pipefail

# Shared MPI-launch helpers (resolve the path before any cd).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/abinit_parallel.sh"

workdir="${1:-}"
if [[ -z "$workdir" ]]; then
  echo "usage: run_relax_wsl.sh <workdir> [--dry-run]" >&2
  exit 2
fi
dry_run=0
case "${2:-}" in
  --dry-run) dry_run=1 ;;
esac

if [[ ! -d "$workdir" ]]; then
  echo "ERROR: directory not found: $workdir" >&2
  echo "  Run from the QuadrupolarDFT root, or pass the path you gave to" >&2
  echo "  'efg_temperature.py relax --out ...' (it is relative to that CWD)." >&2
  exit 1
fi
workdir="$(cd "$workdir" && pwd)"
cd "$workdir"

if ! command -v abinit >/dev/null 2>&1; then
  echo "ERROR: 'abinit' not on PATH (run inside WSL, not Git Bash)." >&2
  exit 1
fi
export ABI_PSPDIR="${ABI_PSPDIR:-/usr/share/abinit/psp}"

if [[ ! -f relax.abi ]]; then
  echo "ERROR: $workdir/relax.abi not found." >&2
  echo "  Stage it first: efg_temperature.py relax --base <static>.abi --out $workdir" >&2
  exit 1
fi

# Resolve the launch command (serial, or MPI per ABINIT_NP=<N>|auto).
abinit_build_cmd relax.abi || exit 1

echo "==> ABINIT ionic relaxation in $workdir"
extra=()
(( dry_run )) && extra=(--dry-run)
# Stream to the terminal AND capture to relax.stdout/relax.stderr.
if ! "${abinit_cmd[@]}" relax.abi "${extra[@]}" > >(tee relax.stdout) 2> >(tee relax.stderr >&2); then
  echo "ERROR: ABINIT exited non-zero. Last lines of relax.stderr:" >&2
  tail -n 20 relax.stderr >&2 || true
  echo "(The relaxation input is a starting template -- check tolmxf/ntime/k-mesh.)" >&2
  exit 1
fi

if (( dry_run )); then
  echo "Dry run only."
  exit 0
fi

# ABINIT names the main output after the input by default (relax.abo). Some
# builds/configs write relax.out or relax.abi.abo instead; accept any of them.
abo=""
for cand in relax.abo relax.out relax.abi.abo; do
  [[ -f "$cand" ]] && { abo="$cand"; break; }
done
if [[ -z "$abo" ]]; then
  echo "ERROR: no relax.abo/relax.out written; check relax.stderr above." >&2
  exit 1
fi
[[ "$abo" != "relax.abo" ]] && cp "$abo" relax.abo

if grep -q "outvars: echo values of variables after computation" relax.abo; then
  echo "Done. Relaxed geometry in $workdir/relax.abo"
  echo "Next: efg_temperature.py relax-collect --base <static>.abi --abo $workdir/relax.abo --out $workdir"
else
  echo "ERROR: relax.abo has no post-computation footer; the run did not finish." >&2
  echo "  (No relaxed geometry to read -- inspect relax.abo for SCF/relax errors.)" >&2
  exit 1
fi
