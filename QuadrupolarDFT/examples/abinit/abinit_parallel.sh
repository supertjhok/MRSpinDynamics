#!/usr/bin/env bash
# Shared MPI-launch helpers for the ABINIT runner scripts.
#
# Source this file, then call `abinit_build_cmd <input.abi>` to populate the
# global array `abinit_cmd` with the right way to launch ABINIT:
#
#   ABINIT_NP unset or 1 -> serial:    abinit
#   ABINIT_NP=<N>        -> parallel:  mpirun -np N abinit
#   ABINIT_NP=auto       -> parallel:  N chosen automatically (see below)
#
# Override the launcher with ABINIT_MPIRUN, e.g. "mpiexec", or
# "mpirun --oversubscribe", or "mpirun --allow-run-as-root" if your WSL user is
# root. These functions do not exit the calling script; on a launcher error
# `abinit_build_cmd` returns non-zero and leaves `abinit_cmd` serial.

# Largest divisor of nkpt ($1) that is <= the core count ($2). Echoes the result.
#
# ABINIT's default k-point parallelism (paral_kgb 0) is efficient when the number
# of MPI processes divides the number of k-points in the IBZ, so the best
# balanced choice is the largest such divisor that fits in the available cores.
abinit_optimal_np() {
  local nkpt="${1:-0}" maxnp="${2:-1}" best=1 d
  if [[ ! "$nkpt" =~ ^[0-9]+$ ]] || (( nkpt < 1 )); then echo 1; return; fi
  if [[ ! "$maxnp" =~ ^[0-9]+$ ]] || (( maxnp < 1 )); then maxnp=1; fi
  for (( d = 1; d <= nkpt && d <= maxnp; d++ )); do
    if (( nkpt % d == 0 )); then best=$d; fi
  done
  echo "$best"
}

# Echo the IBZ k-point count (nkpt) for an ABINIT input or output file.
# If the file already contains an "nkpt" line (e.g. a .abo), read it; otherwise
# run a fast `abinit <file> --dry-run` to compute it without the SCF. Returns
# non-zero if nkpt cannot be determined.
abinit_detect_nkpt() {
  local f="${1:-}" line nkpt=""
  [[ -f "$f" ]] || return 1
  # `^\s*nkpt\s` matches the dimension/echo line but not ngkpt or nkptgw.
  line="$(grep -m1 -iE '^[[:space:]]*nkpt[[:space:]]' "$f" 2>/dev/null || true)"
  nkpt="$(printf '%s' "$line" | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -z "$nkpt" ]]; then
    command -v abinit >/dev/null 2>&1 || return 1
    local tmp; tmp="$(mktemp -d 2>/dev/null)" || return 1
    cp "$f" "$tmp/probe.abi" 2>/dev/null || { rm -rf "$tmp"; return 1; }
    # The input resolves pseudopotentials via $ABI_PSPDIR, so export it here too.
    ( cd "$tmp" && ABI_PSPDIR="${ABI_PSPDIR:-/usr/share/abinit/psp}" \
        abinit probe.abi --dry-run >probe.out 2>&1 ) || true
    line="$(grep -m1 -hiE '^[[:space:]]*nkpt[[:space:]]' "$tmp"/probe.abo "$tmp"/probe.out 2>/dev/null || true)"
    nkpt="$(printf '%s' "$line" | grep -oE '[0-9]+' | head -1 || true)"
    rm -rf "$tmp"
  fi
  [[ -n "$nkpt" ]] || return 1
  echo "$nkpt"
}

# Populate the global array `abinit_cmd`. Arg 1 (optional) is an input/output
# file used to auto-detect nkpt when ABINIT_NP=auto.
abinit_build_cmd() {
  local probe="${1:-}" np="${ABINIT_NP:-1}" from_auto=0
  abinit_cmd=(abinit)
  if [[ "$np" == "auto" ]]; then
    from_auto=1
    local cores nkpt
    cores="$(nproc 2>/dev/null || echo 1)"
    if nkpt="$(abinit_detect_nkpt "$probe")"; then
      np="$(abinit_optimal_np "$nkpt" "$cores")"
      echo "Auto-parallel: nkpt=$nkpt, cores=$cores -> -np $np" >&2
    else
      echo "WARNING: ABINIT_NP=auto but could not determine nkpt; running serial." >&2
      np=1
    fi
  fi
  if [[ "$np" =~ ^[0-9]+$ && "$np" -gt 1 ]]; then
    local launcher
    read -r -a launcher <<< "${ABINIT_MPIRUN:-mpirun}"
    if ! command -v "${launcher[0]}" >/dev/null 2>&1; then
      # auto does its best -> fall back to serial; an explicit N is a hard error.
      if (( from_auto )); then
        echo "WARNING: '${launcher[0]}' not on PATH; running serial." >&2
        return 0
      fi
      echo "ERROR: ABINIT_NP=$np set but '${launcher[0]}' is not on PATH (install an MPI runtime)." >&2
      return 1
    fi
    abinit_cmd=("${launcher[@]}" -np "$np" abinit)
    echo "Parallel: ${abinit_cmd[*]}" >&2
  fi
  return 0
}
