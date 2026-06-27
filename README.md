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
- `References/` is a local, ignored source-material archive used during
  development. It is not committed to Git because it contains copied reference
  documents and large source captures.

Each subproject has its own README or documentation folder with setup and usage
details. Start with `PythonSpinDynamics/` for simulation work, `QuadrupolarDFT/`
for ab initio tensor analysis, and `NQRDatabase/` for spectra data.

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
