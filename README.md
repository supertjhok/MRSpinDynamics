<p align="center">
  <img src="docs/assets/mr_spin_dynamics_logo.svg" alt="MRSpinDynamics: NMR, NQR, and ESR simulations in inhomogeneous fields" width="760">
</p>

# MRSpinDynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21016177.svg)](https://doi.org/10.5281/zenodo.21016177)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

MRSpinDynamics is an open research workspace for magnetic-resonance simulation,
instrument modeling, quadrupolar-parameter analysis, and NQR data curation. It
connects experiment descriptions to spin dynamics, spatial fields, transport,
receiver response, analysis, and reproducible result records instead of treating
those pieces as unrelated scripts.

The Python package is the main development surface. The MATLAB implementation
remains a numerical reference, while the DFT, database, and integration projects
connect calculated electric-field gradients to predicted and measured NQR lines.

## What You Can Model

| Area | Current capabilities |
| --- | --- |
| NMR and imaging | CPMG and FID families; finite echo trains; inversion recovery; phase- and frequency-encoded imaging; field-map and moving-isochromat workflows |
| Sequence design | Backend-neutral RF/gradient/ADC intermediate representation; Pulseq 1.4/1.5 import; signed Pulseq 1.5.0 export; compilation; aligned timeline plots |
| Diffusion and flow | Deterministic PGSE attenuation; seeded 2-D random walkers; restricted boundaries; uniform advective flow; pipe washout and transit-time polarization |
| NQR and ESR/EPR | Selective pulsed NQR, powder averaging, SLSE and population transfer; CW and pulsed ESR, strain, hyperfine, DEER, ESEEM, HYSCORE, and ENDOR models |
| Coupled spins and relaxation | Small dense scalar-coupled systems; exchange; Redfield/dipolar relaxation; radiation damping; semiclassical spin noise |
| Fields and hardware | Coil and magnetostatic fields; PEEC, eddy-current, probe, receiver-noise, RFI rejection, SQUID/OPM, thermal, and electro-thermal models |
| Analysis and optimization | Inverse Laplace tools, parameter sweeps, pulse and gradient optimization, NumPy/Numba/JAX execution paths, and benchmark scaffolds |
| Predict-to-measurement NQR | DFT EFG conversion, Hamiltonian cross-checks, curated spectra with provenance, and predicted-to-observed line matching |

The models intentionally span several levels of description. Bloch isochromats
are used for independent spin-1/2 ensembles, small dense Hamiltonians for
explicit quantum effects, and specialized stochastic or continuum models where
motion, relaxation, fields, or hardware must be represented. The
[Python user manual](PythonSpinDynamics/docs/user_manual.pdf) documents the
boundaries and assumptions for each layer.

## Quick Start: Python Experiments

Python 3.10 or newer is required. From the repository root:

```bash
cd PythonSpinDynamics
python -m pip install -e ".[opt,plot]"
```

The unified experiment facade is the recommended starting point. Describe the
experiment, inspect the resolved plan, then run and save it:

```python
from spin_dynamics.experiment import Experiment, PGSE, Sample

study = Experiment(
    sequence=PGSE(
        gradient_amplitude=0.05,
        gradient_duration=2e-3,
        diffusion_time=20e-3,
    ),
    sample=Sample(diffusion_coefficient=2.1e-9, t2_seconds=0.25),
)

print(study.plan().report())
record = study.run()
record.save("results/pgse_run.npz")
```

TOML and JSON configurations use the same plan/run path:

```bash
python -m spin_dynamics.experiment plan examples/experiment_config_pgse.toml
python -m spin_dynamics.experiment run examples/experiment_config_pgse.toml \
  -o results/pgse_run.npz
python -m spin_dynamics.experiment show results/pgse_run.npz
```

For explicit transport, start with
`examples/experiment_config_pgse_walkers_flow.toml`. Its seeded random walkers
combine diffusion with a uniform 2-D velocity and can be reproduced through the
same CLI.

## Sequence Interchange and Visualization

The sequence layer uses the open [Pulseq](https://pulseq.github.io/) block model:
RF, three gradient channels, and ADC may overlap inside sequential blocks.
Imported sequences can be compiled into piecewise-constant backend inputs or
plotted as aligned lanes with block boundaries and receive samples.

```python
from spin_dynamics.sequences import compile_sequence, read_pulseq

sequence = read_pulseq("protocol.seq")
compiled = compile_sequence(sequence, system_frequency_hz=42.58e6)
figure, axes = compiled.plot(time_unit="ms")
figure.savefig("results/protocol_timeline.png", dpi=150)
```

The bundled script makes timeline inspection a natural part of working with a
sequence file:

```bash
# Plot a built-in spin echo and export it as signed Pulseq 1.5.0.
python examples/plot_sequence_timeline.py \
  --export-pulseq results/demo_spin_echo.seq \
  --output results/demo_spin_echo.png

# Inspect an existing .seq file.
python examples/plot_sequence_timeline.py protocol.seq \
  --system-frequency-hz 42580000 \
  --output results/protocol_timeline.png
```

Pulseq core blocks, RF, arbitrary/default-raster gradients, trapezoids, ADC,
and compressed shapes are supported. Required extensions and explicit
non-default time shapes fail clearly; optional extensions are retained as
metadata but are not yet executed. Export requires raster-aligned timing and
does not silently round the sequence.

## Repository Map

- `MATLABSpinDynamics/` is the original MATLAB implementation. It remains the
  reference point for validated Bloch-equation NMR workflows and historical
  examples. Many core routines also run in GNU Octave; optimization, MATLAB
  Coder/MEX, `parfor`, and some graphics/toolbox workflows still require
  MATLAB or small script edits.
- `PythonSpinDynamics/` is the Python package. It contains validated ports,
  newer simulation engines, sequence interchange, automated tests, examples,
  generated API documentation, and the unified `spin_dynamics.experiment`
  facade. The facade currently resolves NMR, imaging, diffusion, random-walker
  transport, NQR, and ESR specifications onto the underlying workflows.
- `QuadrupolarDFT/` analyzes electric-field-gradient tensors from
  first-principles calculations. These tensors determine nuclear quadrupole
  coupling constants, which are central to NQR interpretation.
- `NQRDatabase/` builds a curated NQR spectra database. It exports SQLite and
  JSONL files, preserves source provenance, links measurements to citations,
  and includes a review workflow for OCR-derived Landolt-Bornstein tables.
- `integration/` is the cross-project layer (`mr_integration`). It connects the
  three subprojects into a single predict-simulate-validate loop: it converts
  ab initio EFG/`C_Q` values into spin-dynamics NQR sites, checks the two
  Hamiltonian implementations against each other, and compares predicted lines
  against the measured database. Its target survey ranks structure-backed
  compounds by the next missing DFT or database input, while its uncertainty
  layer propagates explicit `(C_Q, eta)` distributions into simulated line
  intervals. See `docs/roadmap.md` for the workspace-level survey and plan.
- `References/` is mostly a local, ignored source-material archive used during
  development. Published papers, books, copied reference documents, and large
  source captures should not be committed. The folder does track a small number
  of self-authored technical notes that are useful background for the public
  subprojects.

## Documentation and Development

- [Searchable PythonSpinDynamics documentation](https://supertjhok.github.io/MRSpinDynamics/)
- [Documentation guide and document-status map](docs/documentation_guide.md)
- [Python user manual (PDF)](PythonSpinDynamics/docs/user_manual.pdf) and
  [LaTeX source](PythonSpinDynamics/docs/user_manual.tex)
- [Python package README](PythonSpinDynamics/README.md),
  [generated API reference](PythonSpinDynamics/docs/python_api/api_reference.md),
  [validation evidence matrix](PythonSpinDynamics/docs/validation_matrix.md),
  and [historical validation results](PythonSpinDynamics/docs/validation_results.md)
- [Workspace roadmap and capability gaps](docs/roadmap.md) and
  [release process](docs/release_process.md)
- [QuadrupolarDFT guide](QuadrupolarDFT/README.md),
  [NQR database guide](NQRDatabase/README.md), and
  [integration guide](integration/README.md)

For PythonSpinDynamics development and benchmarking, use the repeatable setup in
[`development_environment.md`](PythonSpinDynamics/docs/development_environment.md)
or the `scripts/setup_dev_env.ps1` and `scripts/setup_dev_env_wsl.sh` helpers.
The WSL helper can install CUDA-enabled JAX with `JAX_CUDA=13`.

The local smoke gate mirrors the hosted job:

```bash
cd PythonSpinDynamics
python -m ruff check src tests examples
python docs/generate_api_reference.py
git diff --exit-code docs/python_api/api_reference.md
python docs/generate_validation_matrix.py --check
python -m unittest tests.smoke_tests
```

Run `python -m unittest discover -s tests` before merging numerical changes.
MATLAB or Octave is only needed when regenerating reference fixtures; the Python
package itself has no runtime dependency on either environment.

## NQR Database Sources

The `NQRDatabase/` subproject currently imports or stages data from these local
source collections:

- an earlier online NQR database associated with Case Western Reserve
  University and the University of Florida, captured locally as Google Sites
  HTML files;
- U.S. Navy / Naval Research Laboratory `NQR_Data_Tables` CHM/PDF exports;
- King's College experimental PDF notes for melamine, metformin HCl,
  paracetamol, and a population-transfer method note;
- H. Chihara and N. Nakamura, *Nuclear Quadrupole Resonance Spectroscopy Data*,
  Landolt-Bornstein, Condensed Matter series, edited by K.-H. Hellwege and
  A. M. Hellwege.

Detailed source paths, imported tables, record counts, and citation handling are
documented in `NQRDatabase/README.md`. Individual paper citations are stored in
the database tables `literature_references` and `reference_links`.

## Technical Notes

Two self-authored notes in `References/` are intentionally shared with the
repository:

- [Ab Initio Electric-Field Gradients and Quadrupolar Resonances in Crystals](References/efg_quadrupolar_technical_note.pdf)
  explains how electric-field-gradient tensors from DFT outputs connect to
  quadrupolar coupling constants, asymmetry parameters, and NQR transition
  frequencies. The LaTeX source is tracked beside the PDF.
- [Modeling Pulsed NQR Dynamics: Spin 1, Spin 3/2, and Higher Spins](References/Pulsed_NQR_Spin_Dynamics_Narrative_Rewrite.pdf)
  motivates the reduced two-level and full density-matrix NQR simulation
  regimes used by `PythonSpinDynamics`. The LaTeX source is tracked beside the
  PDF.

## Releases and Citation

MRSpinDynamics is released as a single citable unit: the whole repository is
versioned together, tagged once (`v<version>`), published as one GitHub Release,
and archived on Zenodo for a citable DOI. The current version and per-component
release notes are in [`CHANGELOG.md`](CHANGELOG.md); the process is documented in
[`docs/release_process.md`](docs/release_process.md).

To cite this software, use the metadata in [`CITATION.cff`](CITATION.cff) (GitHub
renders a "Cite this repository" button from it). The archive is on Zenodo under
the concept DOI [10.5281/zenodo.21016177](https://doi.org/10.5281/zenodo.21016177),
which always resolves to the latest release; each tagged release also gets its own
version DOI.

## License

Copyright (C) 2026 Soumyajit Mandal

**Code** is licensed under the **GNU General Public License v3.0** (GPL-3.0); see
the [LICENSE](LICENSE) file for the full text. The Python workspace is a port of,
and therefore a derivative work of, the GPL-licensed MATLAB code, so the same
license applies to code across the repository. Each subproject also carries its own
`LICENSE` copy (`PythonSpinDynamics/`, `QuadrupolarDFT/`, `integration/`,
`NQRDatabase/`) for when it is viewed in isolation.

**Curated data** is licensed separately. The NQR spectra database under
`NQRDatabase/data/` is released under **CC-BY-4.0** — see
[`NQRDatabase/DATA_LICENSE.md`](NQRDatabase/DATA_LICENSE.md). Individual
measurements are facts; CC-BY applies to the original curation and compilation, and
the underlying values were compiled from third-party sources (Landolt-Börnstein,
U.S. Navy/NRL, King's College, and others) and stored as factual values with
citations.

**Third-party crystal structures.** Most CIF files under
`QuadrupolarDFT/structures/` come from proprietary databases (ICSD, CCDC) that
restrict redistribution. They are not covered by the licenses above — see
[`QuadrupolarDFT/structures/README.md`](QuadrupolarDFT/structures/README.md).

This project also bundles one third-party utility,
`MATLABSpinDynamics/Version_3/labelpoints`, which is
distributed under its own BSD 3-Clause license (Copyright (c) 2017, Adam Danz);
see that directory's `license.txt`.
