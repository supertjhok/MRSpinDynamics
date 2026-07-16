# Test Tiers and Change-Aware Selection

This page tells contributors which checks to run after a change. It is about
efficient software verification; it does not define the scientific evidence
level of a model. For that distinction, read the [Validation
guide](python_api/validation.md).

PythonSpinDynamics has a large test surface because analytical limits, external
fixtures, optional backends, examples, and integrated workflows all matter.
Running the complete suite after every edit is neither necessary nor a good
feedback loop, so the repository uses progressively broader tiers.

## Edit loop: impacted tests

From `PythonSpinDynamics`, run:

```powershell
python scripts\run_impacted_tests.py
```

The selector compares the working tree with `HEAD`, reads
`tests/test_groups.json`, and runs:

1. the short smoke suite;
2. test modules in every subsystem group matched by a changed path;
3. any changed `tests/test_*.py` module directly; and
4. `--help` only for example scripts that changed.

The Git query normalizes Windows/WSL line endings, so invoking the selector
from WSL does not misclassify every CRLF working-tree file as changed.

Inspect the selection without running it:

```powershell
python scripts\run_impacted_tests.py --list
```

Force or combine a subsystem group when a conceptual dependency is broader
than the file paths reveal:

```powershell
python scripts\run_impacted_tests.py --group nqr
python scripts\run_impacted_tests.py --group nqr --group relaxation
```

Use `--base origin/main` to include every difference from another Git base.
The impact map is deliberately declarative: when a subsystem gains a new test
module or a cross-subsystem dependency, update `tests/test_groups.json` in the
same change.

## Fast global smoke

```powershell
python -m unittest tests.smoke_tests
```

This tier contains representative calculations and import/workflow checks. It
must stay short. Catalog-wide example CLI checks belong in
`tests.example_tests`, not in smoke; otherwise smoke time grows with the number
of examples.

## Focused explicit validation

For physics changes, run the exact affected modules explicitly as well. For
example:

```powershell
python -m unittest tests.test_nqr_crossover tests.test_nqr
python -m unittest tests.test_relaxation tests.test_relaxation_workflows
```

The impacted selector prints its exact command, so it is easy to rerun or
extend.

## Full validation

The complete suite remains the authoritative CI, pre-push, and release gate:

```powershell
python scripts\run_impacted_tests.py --full
python -m ruff check src tests examples
python docs\generate_api_reference.py
python docs\generate_validation_matrix.py --check
```

The selector and CI use pytest for impacted and full runs so both unittest
classes and pytest-style test functions execute. The intentionally tiny
`tests.smoke_tests` module remains runnable directly with unittest for the
fastest global sanity check.

The branch-coverage and dedicated full Windows lanes use four pytest-xdist
worker processes with
`--dist loadscope`. BLAS/OpenMP threads remain fixed at one, preventing each
pytest worker (and any powder-process child it starts) from multiplying native
library threads. The coverage lane uses pytest-cov, which combines coverage data
from xdist workers before enforcing the existing branch-coverage floor.
Impacted and smoke tests stay serial because their startup cost is already
small.
The FasterCap interoperability module is excluded from distributed collection
and run once afterward because its optional availability probe initializes a
Windows COM server during module collection.

CI additionally runs branch coverage, optional-backend parity, distribution
installation, and performance-regression jobs. Change-aware selection shortens
local iteration; it does not replace full validation before publishing.

## Design principles

- Put cheap analytical identities and invariants next to the changed module.
- Separate fast correctness tests from expensive numerical convergence tests.
- Keep experimental and cross-language fixtures as validation gates, not inner
  edit-loop checks.
- Give slow tests deterministic reduced-size cases where possible.
- Treat a timeout separately from a numerical assertion failure.
- Record new subsystem dependencies in the impact map.
