<p align="center">
  <img src="docs/assets/mr_spin_dynamics_logo.svg" alt="MRSpinDynamics: NMR, NQR, and ESR simulations in inhomogeneous fields" width="760">
</p>

# MRSpinDynamics

This repository contains sibling workspaces for magnetic-resonance spin
dynamics and ab initio quadrupolar-parameter workflows, now spanning the
original NMR workflows, newer quadrupolar NQR and ESR/EPR extensions, and
first-principles electric-field-gradient calculations:

- `MATLABSpinDynamics/` contains the original MATLAB implementation and remains
  the reference for the validated NMR Bloch-workflow behavior.
- `PythonSpinDynamics/` contains the Python port, tests, validation fixtures,
  examples, API documentation, and Python-native NQR/ESR additions.
- `QuadrupolarDFT/` contains the new Python workspace for ab initio EFG,
  quadrupolar-coupling, and NQR-parameter workflows, starting with ABINIT PAW
  output parsing and backend-neutral tensor analysis.
- `NQRDatabase/` contains a curated NQR spectra database build, with normalized
  JSONL exports, an SQLite database, source/provenance tables, and a Landolt
  review workflow.

Each workspace has its own README with setup notes, examples, and more detailed
documentation. Start with the MATLAB README when checking reference behavior,
and start with the Python README when working on the port or running the Python
package.

The repository is kept as a single GitHub project so the MATLAB reference code,
Python implementation, NQR/ESR extension work, ab initio quadrupolar parameter
workflows, and cross-language validation artifacts can evolve together.

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
