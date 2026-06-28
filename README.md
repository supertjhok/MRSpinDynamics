<p align="center">
  <img src="docs/assets/mr_spin_dynamics_logo.svg" alt="MRSpinDynamics: NMR, NQR, and ESR simulations in inhomogeneous fields" width="760">
</p>

# MRSpinDynamics

MRSpinDynamics is a research workspace for magnetic-resonance simulation,
quadrupolar-parameter analysis, and NQR data curation.

The repository brings together several related projects:

- simulating nuclear magnetic resonance (NMR), nuclear quadrupole resonance
  (NQR), and electron spin resonance/electron paramagnetic resonance (ESR/EPR)
  experiments;
- validating a Python spin-dynamics implementation against an older MATLAB
  reference implementation;
- computing electric-field-gradient and quadrupolar-coupling tensors from
  first-principles electronic-structure outputs;
- building a machine-readable NQR spectra database from archived web pages,
  literature tables, and reviewed PDF extracts.

## Repository Map

- `MATLABSpinDynamics/` is the original MATLAB implementation. It remains the
  reference point for validated Bloch-equation NMR workflows and historical
  examples.
- `PythonSpinDynamics/` is the Python package. It contains the port of the
  MATLAB behavior, automated tests, examples, API documentation, and newer NQR
  and ESR/EPR simulation features.
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
  against the measured database. See `docs/roadmap.md` for the workspace-level
  survey and plan.
- `References/` is mostly a local, ignored source-material archive used during
  development. Published papers, books, copied reference documents, and large
  source captures should not be committed. The folder does track a small number
  of self-authored technical notes that are useful background for the public
  subprojects.

Each subproject has its own README or documentation folder with setup and usage
details. Start with `PythonSpinDynamics/` for simulation work, `QuadrupolarDFT/`
for ab initio tensor analysis, and `NQRDatabase/` for spectra data.

For PythonSpinDynamics development and benchmarking, use the persistent
virtual-environment setup documented in
[`PythonSpinDynamics/docs/development_environment.md`](PythonSpinDynamics/docs/development_environment.md).
The package also provides `PythonSpinDynamics/scripts/setup_dev_env.ps1` and
`PythonSpinDynamics/scripts/setup_dev_env_wsl.sh` so Windows and WSL runs use a
repeatable dependency stack. The WSL setup script also supports CUDA-enabled
JAX installation for GPU benchmarks with `JAX_CUDA=13`.

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

## License

Copyright (C) 2026 Soumyajit Mandal

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).
See the [LICENSE](LICENSE) file for the full text. The Python workspace is a port
of, and therefore a derivative work of, the GPL-licensed MATLAB code, so the same
license applies across the repository.

This project bundles one third-party utility,
`MATLABSpinDynamics/SpinDynamicsUpdated/Version_2/labelpoints`, which is
distributed under its own BSD 3-Clause license (Copyright (c) 2017, Adam Danz);
see that directory's `license.txt`.
