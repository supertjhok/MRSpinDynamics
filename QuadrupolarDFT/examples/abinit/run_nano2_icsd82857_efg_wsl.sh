#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"

cd "$repo_dir"
mkdir -p runs/nano2_icsd82857_efg
cp structures/NaNO2/generated/nano2_icsd82857_efg.abi \
  runs/nano2_icsd82857_efg/nano2_icsd82857_efg.abi
cd runs/nano2_icsd82857_efg

run_abinit_live() {
  local input_file="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  shift 3

  : > "$stderr_file"
  echo "Running ABINIT in $PWD"
  echo "Streaming stdout to terminal and $stdout_file"
  echo "Streaming stderr to terminal and $stderr_file"
  echo "Look for ABINIT progress lines such as ITER STEP NUMBER, ETOT, SCF_istep, and converged."

  ABI_PSPDIR=/usr/share/abinit/psp stdbuf -oL -eL abinit "$input_file" \
    "$@" 2> >(tee "$stderr_file" >&2) | tee "$stdout_file"
}

if [[ "$dry_run" == "1" ]]; then
  run_abinit_live \
    nano2_icsd82857_efg.abi \
    nano2_icsd82857_efg_dryrun.stdout \
    nano2_icsd82857_efg_dryrun.stderr \
    --dry-run
  echo "ABINIT dry-run finished in $PWD"
else
  run_abinit_live \
    nano2_icsd82857_efg.abi \
    nano2_icsd82857_efg.stdout \
    nano2_icsd82857_efg.stderr
  echo "ABINIT finished in $PWD"
fi
