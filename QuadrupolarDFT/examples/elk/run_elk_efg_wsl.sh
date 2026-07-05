#!/usr/bin/env bash
# Run an Elk all-electron EFG calculation and report eta / C_Q for a target
# species. Runs in a WSL-native scratch dir (off any OneDrive-synced /mnt/c
# mount, where heavy I/O stalls) and copies EFG.OUT + logs back.
#
# Usage (from the QuadrupolarDFT root, inside WSL):
#   bash examples/elk/run_elk_efg_wsl.sh examples/elk/nano2_efg.elk.in
#   bash examples/elk/run_elk_efg_wsl.sh examples/elk/nano2_efg.elk.in --species 2 --quadmom 0.02044 --out runs/elk_nano2
#
# Requires elk-lapw on PATH (sudo apt install elk-lapw). Elk reads a file named
# exactly `elk.in`, so the template is copied to elk.in in the scratch dir.
#
# Environment overrides:
#   ELK_SCRATCH_DIR  scratch root (must be WSL-native, not /mnt/c).
#   ELK_NP           MPI ranks. Unset/1 = serial (Elk uses OpenMP over all
#                    cores). >1 launches `mpirun -np ELK_NP elk-lapw` and pins
#                    OMP/OpenBLAS to one thread per rank, i.e. pure MPI over
#                    k-points -- the fastest mode here (see README Performance).
#                    Use ELK_NP ~ min(cores, #irreducible k-points).
#
# Convergence watchdog (polls INFO.OUT, reports progress, auto-kills divergence):
#   ELK_POLL           poll interval in seconds (default 20).
#   ELK_DIVERGE_AFTER  after this many SCF loops, a still-large RMS change is
#                      treated as divergence and the run is killed (default 12).
#   ELK_DIVERGE_RMS    the RMS-change threshold for that check (default 1.0). A
#                      healthy SCF is far below 1 within ~10 loops; an
#                      ill-conditioned basis (high rgkmax vs small muffin-tins)
#                      stays in the hundreds and would otherwise burn the whole
#                      timeout for nothing.
#   ELK_TIMEOUT        hard wall-clock cap in seconds (default 5400).
#
# For real speed also switch the reference BLAS/LAPACK to OpenBLAS (README).
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parser="$script_dir/parse_elk_efg.py"

input="${1:-}"
if [[ -z "$input" || ! -f "$input" ]]; then
  echo "usage: run_elk_efg_wsl.sh <elk-input.elk.in> [--species N] [--quadmom Q] [--out DIR]" >&2
  exit 2
fi
shift || true
species=2
quadmom=0.02044
out_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --species) species="$2"; shift 2 ;;
    --quadmom) quadmom="$2"; shift 2 ;;
    --out) out_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v elk-lapw >/dev/null 2>&1; then
  echo "ERROR: elk-lapw not on PATH (run inside WSL; sudo apt install elk-lapw)." >&2
  exit 1
fi

# Launch command: serial (OpenMP) by default, or pure MPI over k-points when
# ELK_NP > 1. Pure MPI (one thread per rank) load-balances k-points across
# ranks and avoids BLAS/OpenMP oversubscription.
elk_np="${ELK_NP:-1}"
launch=(elk-lapw)
if [[ "$elk_np" =~ ^[0-9]+$ && "$elk_np" -gt 1 ]]; then
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: ELK_NP=$elk_np but 'mpirun' is not on PATH." >&2
    exit 1
  fi
  launch=(mpirun -np "$elk_np" elk-lapw)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  echo "Parallel: mpirun -np $elk_np (OMP/OpenBLAS pinned to 1 thread per rank)"
fi

name="$(basename "$input")"; name="${name%.elk.in}"; name="${name%.in}"
input_abs="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
out_dir="${out_dir:-runs/elk_${name}}"
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

scratch="${ELK_SCRATCH_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/elk_efg.XXXXXX")}"
case "$scratch" in
  /mnt/*) echo "WARNING: scratch $scratch is on a Windows mount; expect slow I/O." >&2 ;;
esac
cp "$input_abs" "$scratch/elk.in"

# ---- launch with a convergence watchdog --------------------------------------
# Elk writes one "Loop number" and one "RMS change in Kohn-Sham potential" per
# SCF iteration to INFO.OUT. We poll it to report progress live and to auto-kill
# a run whose SCF is clearly diverging (RMS still large after ELK_DIVERGE_AFTER
# loops) instead of waiting out the hard timeout. A healthy SCF drops the RMS
# change well below 1 within ~10 loops; an ill-conditioned LAPW basis stays in
# the hundreds and never recovers.
poll_s="${ELK_POLL:-20}"
diverge_after="${ELK_DIVERGE_AFTER:-12}"
diverge_rms="${ELK_DIVERGE_RMS:-1.0}"
timeout_s="${ELK_TIMEOUT:-5400}"
info="$scratch/INFO.OUT"

echo "Running Elk on $input  (${launch[*]})"
echo "  scratch: $scratch   results -> $out_dir"
echo "  watchdog: abort if RMS change > ${diverge_rms} after ${diverge_after} loops; hard cap ${timeout_s}s"
echo "  monitor:  tail -f $info"
echo "            grep 'RMS change' $info | tail    # SCF convergence per loop"

( cd "$scratch" && exec "${launch[@]}" > elk_run.log 2>&1 ) &
run_pid=$!

kill_run() {
  kill -TERM "$run_pid" 2>/dev/null    # mpirun forwards the signal to its ranks
  # Backstop for stragglers: match the binary by exact process NAME (-x), never by
  # command line (-f) -- a -f match would also hit any wrapper script whose command
  # line happens to contain "elk-lapw".
  sleep 2; pkill -TERM -x elk-lapw 2>/dev/null
  sleep 1; kill -KILL "$run_pid" 2>/dev/null; pkill -KILL -x elk-lapw 2>/dev/null
}

start_s="$(date +%s)"
status="ok"; last_loop=-1; last_rms="?"
while kill -0 "$run_pid" 2>/dev/null; do
  sleep "$poll_s"
  [[ -f "$info" ]] || continue
  loop="$(grep -c "Loop number" "$info" 2>/dev/null)"; loop="${loop:-0}"
  rms="$(grep "RMS change" "$info" 2>/dev/null | tail -1 | awk -F: '{print $2}' | awk '{print $1}')"
  if [[ "$loop" != "$last_loop" ]]; then
    echo "  [$(date +%H:%M:%S)] SCF loop ${loop}, RMS change ${rms:-?}"
    last_loop="$loop"; last_rms="${rms:-?}"
  fi
  # Divergence: many loops but RMS still large (or non-numeric, e.g. NaN).
  if (( loop >= diverge_after )) && [[ -n "$rms" ]] \
     && awk -v r="$rms" -v t="$diverge_rms" 'BEGIN{ exit !((r+0) > t || index(tolower(r),"nan") > 0) }'; then
    echo "  >>> DIVERGING: RMS change ${rms} still > ${diverge_rms} after ${loop} loops -- aborting." >&2
    kill_run; status="diverged"; break
  fi
  if (( $(date +%s) - start_s > timeout_s )); then
    echo "  >>> TIMEOUT: exceeded ${timeout_s}s at loop ${loop} -- aborting." >&2
    kill_run; status="timeout"; break
  fi
done
wait "$run_pid" 2>/dev/null; rc=$?

for f in EFG.OUT INFO.OUT TOTENERGY.OUT elk_run.log elk.in; do
  [[ -f "$scratch/$f" ]] && cp -f "$scratch/$f" "$out_dir/"
done

if [[ -f "$scratch/EFG.OUT" ]]; then
  echo "EFG.OUT copied to $out_dir/EFG.OUT"
  python3 "$parser" "$scratch/EFG.OUT" --species "$species" --quadmom "$quadmom"
  echo "Scratch retained at $scratch"
elif [[ "$status" == "diverged" ]]; then
  echo "ERROR: SCF diverged and was auto-killed at loop ${last_loop} (RMS change ${last_rms}" >&2
  echo "  stayed > ${diverge_rms}). Likely an ill-conditioned LAPW basis -- lower rgkmax or" >&2
  echo "  enlarge the muffin-tin radii. Scratch kept at $scratch." >&2
  exit 3
elif [[ "$status" == "timeout" ]]; then
  echo "ERROR: hard timeout (${timeout_s}s) at loop ${last_loop}; SCF may still be converging." >&2
  echo "  Raise ELK_TIMEOUT, add MPI ranks (ELK_NP), or enable OpenBLAS. Scratch: $scratch." >&2
  exit 124
else
  echo "ERROR: Elk exited (rc=$rc) without EFG.OUT. Diagnostic:" >&2
  grep -iE "Error|STOP|too small|overlap" "$scratch/elk_run.log" 2>/dev/null | head -6 >&2
  echo "(Scratch kept at $scratch for inspection.)" >&2
  exit 1
fi
