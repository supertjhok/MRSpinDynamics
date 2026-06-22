<p align="center">
  <img src="docs/assets/mr_spin_dynamics_logo.svg" alt="MRSpinDynamics: NMR, NQR, and ESR simulations in inhomogeneous fields" width="760">
</p>

# MRSpinDynamics

This repository contains two sibling workspaces for magnetic-resonance spin
dynamics, now spanning the original NMR workflows and the newer quadrupolar NQR
and ESR/EPR extensions:

- `MATLABSpinDynamics/` contains the original MATLAB implementation and remains
  the reference for the validated NMR Bloch-workflow behavior.
- `PythonSpinDynamics/` contains the Python port, tests, validation fixtures,
  examples, API documentation, and Python-native NQR/ESR additions.

Each workspace has its own README with setup notes, examples, and more detailed
documentation. Start with the MATLAB README when checking reference behavior,
and start with the Python README when working on the port or running the Python
package.

The repository is kept as a single GitHub project so the MATLAB reference code,
Python implementation, NQR/ESR extension work, and cross-language validation
artifacts can evolve together.
