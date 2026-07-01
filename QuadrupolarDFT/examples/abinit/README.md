# ABINIT Examples

These examples are intended to start reproducible EFG calculations, not to
provide converged production inputs.

## Relaxed-vs-unrelaxed temperature study

`nano2_relaxation_study.sh` compares the finite-temperature EFG of the relaxed
geometry against the unrelaxed one, holding everything but the geometry fixed. By
default it runs both branches with identical settings and writes a side-by-side
comparison against the measured NaNO2 ¹⁴N reference; pass
`--reuse-unrelaxed <dir>` to reuse an existing unrelaxed run instead of
recomputing it, and `--study-dir <dir>` to choose where the study is written. See
[`README_relaxation_study.md`](README_relaxation_study.md) for the objective,
what is held fixed, prerequisites, the one-command run, the stage-by-stage
fallback, and how to read the report. Quick start (inside WSL, from the
`QuadrupolarDFT` root):

```bash
bash examples/abinit/nano2_relaxation_study.sh --dry-run   # validate inputs
bash examples/abinit/nano2_relaxation_study.sh             # full study
```

## `nano2_efg.abi`

`nano2_efg.abi` is a starter ABINIT PAW calculation for ferroelectric sodium
nitrite, NaNO2. It is meant to produce nonzero quadrupolar couplings for
NQR-relevant nuclei and to exercise the parser/analysis workflow.

Important caveats:

- The file uses an explicit low-symmetry conventional-cell geometry so ABINIT
  does not need to reconstruct the `Im2m` setting.
- The cell and internal coordinates are a hand-entered starter geometry. Replace
  them with a preferred experimental CIF before interpreting absolute `C_Q`
  values.
- The run uses the `Pseudodojo_paw_pbe_standard` PAW datasets shipped with the
  Ubuntu ABINIT package.
- `ecut`, `pawecutdg`, and `ngkpt` are initial values. Converge EFG tensor
  components, not only total energy.

The structural context is the ferroelectric sodium-nitrite phase discussed by
Fokin et al., arXiv:cond-mat/0205303, which reports `Im2m` below the
ferroelectric transition and `Immm` above it.

## `run_nano2_icsd82857_efg_wsl.sh`

Runs the ABINIT input generated from ICSD 82857:
`structures/NaNO2/generated/nano2_icsd82857_efg.abi`.

Dry-run validation from the `QuadrupolarDFT` directory:

```bash
bash examples/abinit/run_nano2_icsd82857_efg_wsl.sh --dry-run
```

Full run:

```bash
bash examples/abinit/run_nano2_icsd82857_efg_wsl.sh
```

The run scripts stream ABINIT output to the terminal while saving it to
`runs/<case>/*.stdout` and `runs/<case>/*.stderr`. During SCF work, ABINIT
prints progress markers such as `ITER STEP NUMBER`, `ETOT`, `SCF_istep`, and
`converged`.

Post-process a completed ICSD 82857 run into the tracked Markdown and CSV result
files with:

```bash
PYTHONPATH=src python3 examples/postprocess_nano2_efg.py \
  --run-dir runs/nano2_icsd82857_efg \
  --case-id nano2_icsd82857_efg \
  --title "NaNO2 ICSD 82857 ABINIT EFG Run" \
  --input structures/NaNO2/generated/nano2_icsd82857_efg.abi \
  --analysis-date 2026-06-26 \
  --note "This calculation uses the ICSD 82857 experimental ferroelectric NaNO2 structure expanded into the conventional 8-atom cell."
```

Run the ICSD 82857 case from the repository root with:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/QuadrupolarDFT" && bash examples/abinit/run_nano2_icsd82857_efg_wsl.sh'
```

Run the original starter case from the repository root with:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/QuadrupolarDFT" && bash examples/abinit/run_nano2_efg_wsl.sh'
```

Validate without launching the SCF with:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/QuadrupolarDFT" && bash examples/abinit/run_nano2_efg_wsl.sh --dry-run'
```

or manually in WSL:

```bash
cd "/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/QuadrupolarDFT"
mkdir -p runs/nano2_efg
cp examples/abinit/nano2_efg.abi runs/nano2_efg/nano2_efg.abi
cd runs/nano2_efg
ABI_PSPDIR=/usr/share/abinit/psp stdbuf -oL -eL abinit nano2_efg.abi \
  2> >(tee nano2_efg.stderr >&2) | tee nano2_efg.stdout
```

## Glycine Strain-To-EFG Coupling

`strain_efg_coupling.py` stages homogeneous-strain EFG calculations and collects
the completed ABINIT outputs into transition-drive couplings in `Hz/strain`.
The default CIF check points at bundled glycine CCDC 189379 and verifies that it
is the non-centrosymmetric `P 21` polymorph before staging piezoelectric jobs.

From the `QuadrupolarDFT` root:

```bash
PYTHONPATH=src python3 examples/abinit/strain_efg_coupling.py check

PYTHONPATH=src python3 examples/abinit/strain_efg_coupling.py prepare-base \
  --out runs/glycine_static/glycine_efg.abi

bash examples/abinit/run_static_efg_wsl.sh \
  runs/glycine_static/glycine_efg.abi

PYTHONPATH=src python3 examples/abinit/strain_efg_coupling.py generate \
  --base runs/glycine_static/glycine_efg.abi \
  --target-atom-index 1 \
  --out runs/glycine_strain

bash examples/abinit/run_strain_efg_wsl.sh runs/glycine_strain

PYTHONPATH=src python3 examples/abinit/strain_efg_coupling.py collect \
  --workdir runs/glycine_strain \
  --quadmom 0.02044 \
  --strain-peak 1e-5 \
  --json runs/glycine_strain/couplings.json \
  --csv runs/glycine_strain/couplings.csv
```

Use `--dry-run` on `run_strain_efg_wsl.sh` to ask ABINIT to validate all staged
inputs without running the full SCF:

```bash
bash examples/abinit/run_strain_efg_wsl.sh runs/glycine_strain --dry-run
```

`--target-atom-index` is zero-based.  The generated glycine base input expands
the `P 21` symmetry operations; for the bundled CIF, `prepare-base` reports two
equivalent nitrogen candidates, indices `1` and `11`.

The Ubuntu Noble ABINIT package ships glycine-compatible H/C/N/O PAW XML files
in `Pseudodojo_paw_pw_standard`, but those datasets can stop on PAW-sphere
overlap for short X-H bonds in molecular crystals.  For exploratory runs, you
can deliberately allow overlap with, for example:

```bash
PYTHONPATH=src python3 examples/abinit/strain_efg_coupling.py prepare-base \
  --out runs/glycine_static/glycine_efg.abi \
  --pawovlp 25
```

Treat this as a workaround, not a converged production setting.  For publishable
EFGs, prefer validating against smaller-radius PAW datasets or an ABINIT build
with a vetted pseudopotential set.
