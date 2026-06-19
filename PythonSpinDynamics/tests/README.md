# Tests

Tests should compare Python outputs against small MATLAB reference cases before
the Python API is treated as stable.

Initial validation targets:

1. `calc_time_domain_echo.m` on a small synthetic complex spectrum.
2. `sim_spin_dynamics_asymp_mag3.m` using the ideal CPMG quick-start inputs.
3. `sim_spin_dynamics_arb10.m` using the existing MATLAB benchmark parameters.
4. Probe-specific CPMG examples after the ideal path is stable.

Keep reference arrays small enough to commit as text or NumPy `.npz` fixtures.

## Test Tiers

Run the fast smoke tier during normal edit loops:

```powershell
python -m unittest tests.smoke_tests
```

Run the full MATLAB/Octave fixture validation before committing changes that
touch numerical behavior or public workflows:

```powershell
python -m unittest tests.fixture_tests
```

Run the example-script tier after changing public examples or example helpers:

```powershell
python -m unittest tests.example_tests
```

Run everything, including skipped validation-gap placeholders:

```powershell
python -m unittest discover -s tests
```

Install development dependencies before running lint or optional-dependency
validation:

```powershell
python -m pip install -e ".[dev,opt,plot,bench]"
python -m ruff check src tests examples
```

The smoke tier intentionally samples representative fixture, workflow, pulse,
and example checks. It is not a replacement for full validation. The full suite
is grouped into smaller fixture modules behind `test_basic_octave_fixtures.py`
so existing commands still work while failures remain easier to localize.

`test_validation_gaps.py` contains skipped tests for known scientific
validation gaps. Those skips are expected until the corresponding MATLAB or
historical reference data exists.

Matched-probe SPA summary checks are intentionally kept out of the smoke tier
because each selected catalog pulse runs the matched-network transient solver.
