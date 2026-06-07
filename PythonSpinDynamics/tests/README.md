# Tests

Tests should compare Python outputs against small MATLAB reference cases before
the Python API is treated as stable.

Initial validation targets:

1. `calc_time_domain_echo.m` on a small synthetic complex spectrum.
2. `sim_spin_dynamics_asymp_mag3.m` using the ideal CPMG quick-start inputs.
3. `sim_spin_dynamics_arb10.m` using the existing MATLAB benchmark parameters.
4. Probe-specific CPMG examples after the ideal path is stable.

Keep reference arrays small enough to commit as text or NumPy `.npz` fixtures.
