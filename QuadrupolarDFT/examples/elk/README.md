# Elk all-electron EFG cross-check

These scripts run an **all-electron** (full-potential LAPW) electric-field-gradient
calculation with the [Elk code](https://elk.sourceforge.io/) as an independent
check on the ABINIT PAW pipeline. All-electron LAPW is the reference standard for
EFGs, so it isolates whether a disagreement with experiment comes from the
*functional* or from the *pseudopotential/PAW* treatment near the nucleus.

## Why this exists — the NaNO2 result

For the ¹⁴N site of ferroelectric NaNO₂ (ICSD 82857 geometry), the asymmetry
parameter `eta` comes out very differently between PAW and all-electron with the
**same PBE functional and the same geometry**:

| method (PBE, ICSD geometry) | eta | \|C_Q\| (MHz) |
|---|---|---|
| Experiment (77–300 K NQR) | **0.38** | 5.49 |
| ABINIT — PAW (Pseudodojo PBE) | 0.112 | 5.03 |
| Elk — all-electron LAPW | **≈0.34** | ≈5.9 |

All-electron PBE recovers `eta ≈ 0.34`, next to experiment, while the PAW result
sits ~3× lower. The transverse EFG components that set `eta` are a near-nucleus
quantity, and the Pseudodojo PBE **N** dataset under-represents them. So the long
`eta` shortfall seen across the ABINIT study was largely a **PAW/pseudopotential
artifact, not a functional deficiency** — plain PBE is essentially right once the
core region is treated all-electron. (Convergence of the Elk number is checked in
the table below.)

## Install

```bash
sudo apt install elk-lapw     # provides /usr/bin/elk-lapw + species files
```

The package ships all element species files in `/usr/share/elk-lapw/species/`.

## Run

From the `QuadrupolarDFT` root, inside WSL:

```bash
bash examples/elk/run_elk_efg_wsl.sh examples/elk/nano2_efg.elk.in --species 2
```

The runner copies the input to `elk.in` in a WSL-native scratch dir (off any
OneDrive-synced `/mnt/c` mount, where heavy I/O stalls), runs `elk-lapw`, copies
`EFG.OUT`/`INFO.OUT`/logs back to `runs/elk_nano2/`, and prints `eta`, `Vzz`, and
`C_Q` for the target species via `parse_elk_efg.py`. Nitrogen is **species 2** in
`nano2_efg.elk.in` (the order of the `atoms` block). To read a completed run
directly:

```bash
python3 examples/elk/parse_elk_efg.py runs/elk_nano2/EFG.OUT --species 2 --quadmom 0.02044
```

## Elk gotchas (learned the hard way)

- **`lmaxi 2` is required.** The EFG is an l=2 quantity; the default `lmaxi=1`
  makes task 115 stop with `Error(writeefg): lmaxi too small`.
- **`tasks: 0` then `115`.** Task 115 needs the converged ground state first.
- **`scale 1.88972598858`** lets `avec` be written in Ångström (Å→Bohr).
- **Muffin-tin auto-reduction.** The short nitrite N–O bond (~1.24 Å) forces Elk
  to shrink the N/O muffin-tin radii to ~1.16 Bohr. With small radii, **`rgkmax`
  is the main EFG accuracy knob** — converge it.
- **libxc is version-blocked in the Ubuntu build.** `elk-lapw` 9.2.12 rejects the
  system libxc 5.2.3 (`Error(libxcifc): incompatible Libxc version`), so **all
  libxc functionals are unavailable** — including **r²SCAN** (`xctype 100 497
  498`). These scripts therefore use Elk's **native PBE** (`xctype 3`). Unlocking
  r²SCAN needs a source/conda rebuild of Elk against a compatible libxc; the
  functional (Tier-3) test is deferred behind that. Native functionals (LDA/GGA)
  work fine.

## Performance: OpenBLAS + MPI

All-electron LAPW is diagonalisation-heavy: the dominant cost is the dense
generalised eigenproblem (`zhegv`) solved at every k-point, and it grows steeply
with `rgkmax`. Two settings dominate wall-clock time.

**1. Link a fast BLAS/LAPACK (biggest win).** The Ubuntu `elk-lapw` is linked
against the *reference* BLAS/LAPACK (`/usr/lib/.../blas/libblas.so.3`), whose
`zhegv` is slow and effectively single-threaded -- this is what made the
`rgkmax 9` convergence run crawl (~3.5 min/SCF loop) and time out. Because
`elk-lapw` links the `update-alternatives`-managed `libblas.so.3` /
`liblapack.so.3`, switching the alternative to **OpenBLAS** speeds it up
**without rebuilding Elk** (here it gave ~2x — Elk's OpenMP already parallelised
the k-point loop, so only the per-k eigensolve sped up; the gain is larger when a
single big eigensolve dominates):

```bash
sudo apt install -y libopenblas-dev
sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu     # pick openblas-pthread
sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu   # pick openblas-pthread
```

**2. MPI over k-points.** The packaged `elk-lapw` is an MPI build (it links
`libmpi`), so it parallelises over k-points across ranks. `run_elk_efg_wsl.sh`
exposes this through `ELK_NP`: set it to launch `mpirun -np ELK_NP elk-lapw` with
one thread per rank (pure MPI, no OpenMP/BLAS oversubscription). The useful
number of ranks is bounded by the **irreducible** k-point count, which Elk prints
as `Total number of k-points` in `INFO.OUT` -- for the NaNO2 cell that is 18 for
`ngridk 4 3 3` and 48 for `6 4 4`. So `ELK_NP ~ min(cores, nkpt)`:

```bash
ELK_NP=18 bash examples/elk/run_elk_efg_wsl.sh examples/elk/nano2_efg.elk.in --species 2
```

**Recommended combination** on this 20-core machine: OpenBLAS as the BLAS/LAPACK
alternative, then pure MPI over k-points (`ELK_NP = min(20, nkpt)`, one thread per
rank). This let the dense 6×4×4 run (48 k) finish in reasonable time; note it does
*not* rescue `rgkmax 9`, which diverges (ill-conditioned basis) regardless of
speed. Pure OpenMP (serial launch, `ELK_NP` unset) uses all cores as threads
but leaves the eigensolve on whatever BLAS is selected -- so the OpenBLAS switch
matters in that mode too.

## Convergence watchdog (auto-kill divergent runs)

`run_elk_efg_wsl.sh` monitors the SCF in real time instead of blindly waiting out
the hard timeout. It polls `INFO.OUT` every `ELK_POLL` seconds (default 20),
prints each new SCF loop with its `RMS change in Kohn-Sham potential`, and
**auto-kills the run if the RMS change is still above `ELK_DIVERGE_RMS` (default
1.0) after `ELK_DIVERGE_AFTER` loops (default 12)** — the signature of a diverging
SCF. A hard `ELK_TIMEOUT` (default 5400 s) remains as a backstop.

This matters because the two failure modes look identical from the outside (no
`EFG.OUT` after the timeout) but are completely different:

- **Diverging** (e.g. `rgkmax 9` with the auto-shrunk 1.16 Bohr muffin-tins): the
  RMS change oscillates in the hundreds and never drops — an ill-conditioned LAPW
  basis. The watchdog kills it in ~12 loops and exits **code 3** with a message to
  lower `rgkmax` / enlarge the muffin-tins, instead of burning 90 minutes.
- **Merely slow** (e.g. `ngridk 6 4 4` under reference BLAS): the RMS change marches
  down cleanly toward `1e-6` but there are too many k-points to finish in time.
  That is a *speed* problem — fix it with OpenBLAS + MPI (above), not by giving up.

Exit codes: `0` converged (EFG parsed), `3` diverged (auto-killed), `124` hard
timeout, `1` other Elk error.

## Convergence of the Elk eta (¹⁴N, ICSD geometry)

`rgkmax` sweep (k-mesh 4×3×3 = 18 irreducible k), plus a dense-k check:

| rgkmax | k-mesh | eta | C_Q (MHz) | notes |
|---|---|---|---|---|
| 7 | 4×3×3 | 0.336 | +5.92 | |
| 8 | 4×3×3 | 0.333 | +5.94 | converged; the setting in `nano2_efg.elk.in` |
| 8 | 6×4×4 | 0.333 | +5.94 | dense-k (48 k) — confirms k-convergence (Δeta = 0.0001) |
| 9 | 4×3×3 | — | — | SCF **diverged** (ill-conditioned basis vs the auto-shrunk ~1.16 Bohr muffin-tins); auto-killed by the watchdog |

eta is converged to ~1% in `rgkmax` (7→8) and to 0.0001 in the k-mesh (18→48 k),
so the all-electron value is **eta ≈ 0.333** — versus ABINIT PAW 0.112 and
experiment 0.38.

Convergence note: on the 6×4×4 mesh the SCF enters a mild mixing limit-cycle just
above the default 1e-6 tolerance, so both `epspot` (potential RMS) and `epsengy`
(total energy) were loosened together to 2e-4 (the EFG/eta is fully converged well
before that — the dense-k eta matches the 1e-6-converged coarse-mesh value to
0.0001). Loosen **both**: relaxing only one leaves the other as a silent
bottleneck.

## Files

| file | purpose |
|---|---|
| `nano2_efg.elk.in` | NaNO₂ all-electron PBE EFG input (ICSD geometry) |
| `run_elk_efg_wsl.sh` | run an Elk input in WSL-native scratch, copy results back, parse |
| `parse_elk_efg.py` | parse `EFG.OUT` → `eta`, `Vzz`, `C_Q` for a target species |
