#!/usr/bin/env bash
# Run the DFPT phonon calculation and anaddb for a staged phonon directory.
#
# Usage (run from the QuadrupolarDFT root, or pass an absolute path):
#   bash examples/abinit/run_phonon_wsl.sh runs/nano2_ph
#   bash examples/abinit/run_phonon_wsl.sh runs/nano2_ph --dry-run
#
# Produces in <workdir>:
#   phonon.abo        ABINIT DFPT output (also prints phonon frequencies)
#   *_DDB             derivative database
#   anaddb.out        anaddb output with phonon eigenvectors  <-- feed this to
#                     efg_temperature.py displace --anaddb
#
# NOTE: the anaddb invocation/output layout varies between ABINIT versions; this
# is a best-effort runner for 9.x. If anaddb.out does not parse, fall back to
# efg_temperature.py displace --modes modes.json.
set -uo pipefail

# Shared MPI-launch helpers (resolve the path before any cd).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/abinit_parallel.sh"

workdir="${1:-}"
if [[ -z "$workdir" ]]; then
  echo "usage: run_phonon_wsl.sh <workdir> [--dry-run]" >&2
  exit 2
fi
dry_run=0
anaddb_only=0
case "${2:-}" in
  --dry-run) dry_run=1 ;;
  --anaddb-only) anaddb_only=1 ;;  # reuse an existing DFPT DDB, skip the long run
esac

if [[ ! -d "$workdir" ]]; then
  echo "ERROR: directory not found: $workdir" >&2
  echo "  Run from the QuadrupolarDFT root, or pass the path you gave to" >&2
  echo "  'efg_temperature.py phonon --out ...' (it is relative to that CWD)." >&2
  exit 1
fi
workdir="$(cd "$workdir" && pwd)"
cd "$workdir"

if ! command -v abinit >/dev/null 2>&1; then
  echo "ERROR: 'abinit'/'anaddb' not on PATH (run inside WSL, not Git Bash)." >&2
  exit 1
fi
export ABI_PSPDIR="${ABI_PSPDIR:-/usr/share/abinit/psp}"
# anaddb is always run serially below -- it is cheap and its MPI path is
# version-sensitive; only the ABINIT DFPT step honors ABINIT_NP.

if (( ! anaddb_only )); then
  if [[ ! -f phonon.abi ]]; then
    echo "ERROR: $workdir/phonon.abi not found." >&2
    echo "  Stage it first: efg_temperature.py phonon --base <static>.abi --out $workdir" >&2
    exit 1
  fi
  # Resolve the launch command (serial, or MPI per ABINIT_NP=<N>|auto).
  abinit_build_cmd phonon.abi || exit 1
  echo "==> ABINIT DFPT phonon in $workdir"
  extra=()
  (( dry_run )) && extra=(--dry-run)
  # Stream to the terminal AND capture to phonon.stdout/phonon.stderr.
  if ! "${abinit_cmd[@]}" phonon.abi "${extra[@]}" > >(tee phonon.stdout) 2> >(tee phonon.stderr >&2); then
    echo "ERROR: ABINIT exited non-zero. Last lines of phonon.stderr:" >&2
    tail -n 20 phonon.stderr >&2 || true
    echo "(The DFPT input is a starting template -- check tolerances/k-mesh/PAW DFPT support.)" >&2
    exit 1
  fi
  if (( dry_run )); then
    echo "Dry run only; not running anaddb."
    exit 0
  fi
fi

# Pick the DFPT derivative database: the largest *_DDB (a GS-dataset DDB, e.g.
# *_DS1_DDB, is tiny and has no dynamical matrix; the DFPT one, *_DS2_DDB, is
# much larger). Selecting by size avoids feeding anaddb the empty GS DDB.
ddb="$(ls -1S *_DDB 2>/dev/null | head -1 || true)"
if [[ -z "$ddb" ]]; then
  echo "ERROR: no *_DDB found in $workdir." >&2
  (( anaddb_only )) && echo "  (--anaddb-only needs a completed DFPT run first.)" >&2
  exit 1
fi
echo "==> DDB: $ddb ($(stat -c%s "$ddb" 2>/dev/null || echo '?') bytes)"

cat > anaddb.files <<EOF
anaddb.abi
anaddb.out
$ddb
anaddb_band
anaddb_gkk_in
anaddb_ep
anaddb_ddk
EOF

echo "==> anaddb"
if ! anaddb < anaddb.files > >(tee anaddb.log) 2>&1; then
  echo "ERROR: anaddb exited non-zero. Last lines of anaddb.log:" >&2
  tail -n 20 anaddb.log >&2 || true
  echo "(anaddb invocation is version-sensitive; if stuck, use displace --modes modes.json.)" >&2
  exit 1
fi

if [[ -f anaddb.out ]]; then
  echo "Done. Eigenvectors in $workdir/anaddb.out"
  echo "Next: efg_temperature.py displace --base <static>.abi --anaddb $workdir/anaddb.out --target <i> --out runs/<disp>"
else
  echo "ERROR: anaddb did not write anaddb.out; see anaddb.log above." >&2
  exit 1
fi
