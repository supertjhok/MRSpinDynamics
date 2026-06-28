# NaNO2 ¹⁴N relaxation study (relaxed vs unrelaxed)

This is a runnable, end-to-end study that answers one question: **how much does
relaxing the structure to an energy minimum improve the finite-temperature EFG
relative to the earlier unrelaxed run?**

By default the study is **self-contained**: it runs the full finite-displacement
workflow on both the unrelaxed starter geometry and the relaxed geometry with
identical settings, so only the structure changes. It writes a side-by-side
comparison against the measured NaNO2 ¹⁴N reference. If you already have a
matching unrelaxed run, `--reuse-unrelaxed <dir>` reuses it instead of
recomputing (see *Reusing an existing unrelaxed run*).

## What is held fixed

A clean comparison must change one thing. Both branches use the same target atom,
number of modes, displacement amplitude, temperatures, and `Q`; and the same DFT
settings — `ecut`, `pawecutdg`, `ngkpt`, pseudopotentials, etc. — because the
relaxed input `relaxed.abi` is the original `nano2_efg.abi` with only its atomic
positions replaced.

The one thing that does change besides the geometry is the phonon spectrum — the
relaxed structure has different (and, ideally, no longer imaginary) modes. That
is the physics under test, not a confound: the **static EFG** rows of the report
isolate the pure geometry effect (η, line splitting), independent of the modes.

## What "good" looks like

The unrelaxed starter geometry is known to be wrong in specific ways (see the
user manual, *Worked Example* caveat): its static `η ≈ 0.04` disagrees with the
experimental `0.38`, and the two strong ¹⁴N lines come out near-degenerate
(`~3.7 MHz`) instead of well split (`3.60 / 4.64 MHz`). Relaxation should move all
three toward experiment:

| quantity | unrelaxed (starter) | measured (ordered phase) |
|---|---|---|
| `η` (static) | ~0.04 | ~0.38 |
| strong lines | ~3.7, ~3.7 MHz | 3.60, 4.64 MHz |
| `C_Q` | ~5.2 MHz | ~5.49 MHz |
| `dν/dT` sign | negative (Bayer) | negative |

The measured numbers come from the NQR-database ¹⁴N series used elsewhere in the
workspace (`integration/examples/nano2_temperature_coefficients.py`): `ν₊` from
4929 kHz (77 K) to 4637 kHz (300 K), `ν₋` from 3757 kHz to 3601 kHz.

## Prerequisites

- **WSL** (e.g. Ubuntu) with `abinit` **and** `anaddb` on `PATH`. The runner
  scripts refuse to start otherwise. Run everything from inside WSL, not Git Bash.
- The PBE PAW datasets referenced by `nano2_efg.abi`
  (`$ABI_PSPDIR/Pseudodojo_paw_pbe_standard/{Na,N,O}.xml`; the scripts default
  `ABI_PSPDIR=/usr/share/abinit/psp`).
- No install needed — the driver sets `PYTHONPATH=src` itself.

## Run it (one command)

From the `QuadrupolarDFT` root, inside WSL:

```bash
# Validate the inputs first (stages the relaxation + phonon inputs and runs
# `abinit --dry-run` on them; no SCF, takes seconds):
bash examples/abinit/nano2_relaxation_study.sh --dry-run

# Then the full study. By default both branches run DFT (a DFPT phonon run plus
# ~2·max_modes+1 EFG runs each), so this takes a while:
bash examples/abinit/nano2_relaxation_study.sh

# Put the study somewhere of your choosing:
bash examples/abinit/nano2_relaxation_study.sh --study-dir runs/my_study
```

From Windows PowerShell you can launch it in one shot:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/QuadrupolarDFT" && bash examples/abinit/nano2_relaxation_study.sh'
```

Useful flags: `--study-dir DIR` (where the study is written; default
`runs/nano2_relax_study`), `--base FILE`, `--target` (0-based resonant atom,
default `2` = first N), `--max-modes N`, `--max-displacement Å`,
`--temperatures 0,77,150,300`, and `--reuse-unrelaxed DIR` (below).

### Parallelization (automatic)

Serial ABINIT on this 36-k-point cell is slow — the ground-state SCF alone can
take a couple of hours per branch. ABINIT is MPI-built, so the study **picks the
number of processes for you**: by default (`ABINIT_NP=auto`) it reads your core
count and the run's k-point count and launches `mpirun -np N` with the best `N`,
falling back to serial if no MPI is found.

```bash
bash examples/abinit/nano2_relaxation_study.sh            # auto -np (default)
bash examples/abinit/nano2_relaxation_study.sh --np 18    # pin a value
bash examples/abinit/nano2_relaxation_study.sh --np 1     # force serial
```

This changes *only* how ABINIT is launched, not the inputs, so the
relaxed-vs-unrelaxed comparison stays identical. Override the launcher with
`ABINIT_MPIRUN` (e.g. `ABINIT_MPIRUN=mpiexec`, or `"mpirun --oversubscribe"`, or
`"mpirun --allow-run-as-root"` if your WSL user is root).

**How `auto` chooses N.** These runs use ABINIT's default k-point parallelism
(`paral_kgb 0`), which the manual says is efficient *"provided the number of MPI
procs divides the number of k-points in the IBZ."* So the rule, implemented in
`abinit_parallel.sh` (`abinit_optimal_np`), is:

> `N` = the largest divisor of `nkpt` that is ≤ your core count.

`auto` detects `nkpt` by reading an existing `.abo` or, failing that, a quick
`abinit --dry-run`, and the core count from `nproc`. For this cell (`nkpt = 36`,
divisors 2/3/4/6/9/12/18/36) on a 20-core box it selects **N = 18** (2 k-points
per process). The same logic and the `ABINIT_NP`/`ABINIT_MPIRUN` knobs work in the
individual `run_*_wsl.sh` scripts (which all `source abinit_parallel.sh`).

Beyond `N = nkpt` the k-point parallelism saturates; going further needs band/FFT
parallelism (`paral_kgb 1`), not worth it for a cell this small. To let ABINIT
itself profile distributions, add `autoparal 1` + `max_ncpus M` to an input — it
prints an efficiency table for 2…M processes and stops (most useful for larger
cells).

## What the two branches do

**Unrelaxed branch** (default): `phonon → displace → collect` on `nano2_efg.abi`.

**Relaxed branch**: `relax → relax-collect` produces `relaxed.abi` (the starter
input at the relaxed geometry, EFG keywords intact), then `phonon → displace →
collect` on `relaxed.abi`. The relaxation is internal-coordinates-only
(`optcell 0`); the experimental cell is held fixed.

Both branches use the shared `--target`/`--max-modes`/`--max-displacement` and the
same temperatures/Q/spin, write a results JSON (`unrelaxed.json`, `relaxed.json`),
and the driver feeds both to `compare_relaxation.py`.

## Reusing an existing unrelaxed run

If you already ran the unrelaxed geometry, skip recomputing it:

```bash
bash examples/abinit/nano2_relaxation_study.sh --reuse-unrelaxed runs/nano2_disp
```

The driver re-collects that workdir's EFG outputs into `unrelaxed.json` (no
ABINIT) with the study's temperatures/Q/spin, and reads the **displacement
parameters** (target atom, number of modes, amplitude) back from its
`manifest.json`, applying them to the relaxed branch so the two sides stay
identical. There is no default reuse path — you point it at your run.

## Outputs

Everything lands under the study directory (default `runs/nano2_relax_study/`,
git-ignored):

```
unrelaxed.json                 unrelaxed C_Q(T), η(T), lines, dν/dT
relaxed.json                   relaxed-branch results
relaxation_comparison.md       <-- the side-by-side report (read this)
unrelaxed/phonon, unrelaxed/disp   staged inputs + ABINIT outputs
relaxed/relax, relaxed/phonon, relaxed/disp
```

(With `--reuse-unrelaxed` the `unrelaxed/` tree is not created — only
`unrelaxed.json` is written from the run you point at.)

The report (`relaxation_comparison.md`) has four sections: the static (0 K) EFG,
the temperature sweep, the per-line `dν/dT`, and a short "Reading the result"
verdict that states whether `η` moved toward 0.38 and how much the two strong
lines split — each against the measured column.

## Running the stages by hand

If a stage fails midway (DFPT and anaddb are the fragile steps), run the relaxed
branch stage by stage — every step is an ordinary `efg_temperature.py` call:

```bash
export PYTHONPATH=src
EFG=examples/abinit/nano2_efg.abi
D=runs/nano2_relax_study/relaxed

# Stage 0: relax
python3 examples/abinit/efg_temperature.py relax --base $EFG --out $D/relax
bash examples/abinit/run_relax_wsl.sh $D/relax
python3 examples/abinit/efg_temperature.py relax-collect \
    --base $EFG --abo $D/relax/relax.abo --out $D/relax

# Stages 1-3 on the relaxed input
python3 examples/abinit/efg_temperature.py phonon --base $D/relax/relaxed.abi --out $D/phonon
bash examples/abinit/run_phonon_wsl.sh $D/phonon
python3 examples/abinit/efg_temperature.py displace --base $D/relax/relaxed.abi \
    --anaddb $D/phonon/anaddb.out --target 2 --max-modes 6 --out $D/disp
bash examples/abinit/run_finite_displacement_wsl.sh $D/disp
python3 examples/abinit/efg_temperature.py collect --workdir $D/disp \
    --temperatures 0,77,150,300 --quadmom 0.02044 --label relaxed \
    --out-json runs/nano2_relax_study/relaxed.json
```

## Caveats

- **Starter cell.** `nano2_efg.abi` is a hand-entered starter geometry with
  unconverged `ecut`/`pawecutdg`/`ngkpt`. The study shows the *effect of
  relaxation*; absolute agreement also needs a converged cell (ideally from an
  experimental CIF) and converged cutoffs/k-mesh.
- **Templates.** The generated relaxation and DFPT inputs are starting templates
  — verify `tolmxf`/`ntime` (relax) and the DFPT tolerances/k-mesh before
  trusting production numbers.
- **anaddb layout.** If the anaddb eigenvector parser trips on your ABINIT
  version, pass `displace --modes modes.json` instead (see the main README).
- **Residual gap.** Even fully relaxed, a single set of harmonic modes will not
  reproduce NaNO2's full slope near its ferroelectric transition — that softening
  is the case for AIMD/PIMD averaging, noted in the roadmap.
