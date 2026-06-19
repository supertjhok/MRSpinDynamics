<p align="center">
  <img src="docs/assets/nmr_spin_dynamics_logo.svg" alt="NMRSpinDynamics: spin-1/2 simulations in inhomogeneous fields" width="760">
</p>

# NMR Spin Dynamics

This repository contains two sibling workspaces for simulating NMR spin
dynamics:

- `MATLABSpinDynamics/` contains the original MATLAB implementation and remains
  the reference for numerical behavior.
- `PythonSpinDynamics/` contains the Python port, tests, validation fixtures,
  examples, and API documentation.

Each workspace has its own README with setup notes, examples, and more detailed
documentation. Start with the MATLAB README when checking reference behavior,
and start with the Python README when working on the port or running the Python
package.

The repository is kept as a single GitHub project so the MATLAB reference code,
Python implementation, and cross-language validation artifacts can evolve
together.
