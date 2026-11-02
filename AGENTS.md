# Project Memory

- The working checkout is `C:\dev\MRSpinDynamics` (WSL path
  `/mnt/c/dev/MRSpinDynamics`). An older copy under
  `C:\Users\super\OneDrive\Codex\NMR` is stale and is not a git repository; do
  not run or validate anything there.
- For PythonSpinDynamics examples or tests that need SciPy, Matplotlib, or a Linux
  plotting environment, use WSL distribution `Ubuntu-24.04`. It already has
  `numpy`, `scipy`, and `matplotlib` available. Example:
  `wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/dev/MRSpinDynamics/PythonSpinDynamics' && python3 examples/plot_dexsy_exchange.py --output .tmp/dexsy_exchange_wsl.png"`.
- Avoid using `Ubuntu2404Codex` for this repository's plotting/NNLS verification
  unless explicitly requested; it did not have the needed Python packages when
  last checked.
- Before pushing PythonSpinDynamics updates, reproduce the GitHub smoke job, not
  just `python -m unittest tests.smoke_tests`. The smoke workflow
  (`.github/workflows/python-spin-dynamics.yml`) runs, from `PythonSpinDynamics/`:
  1. `python -m ruff check src tests examples`
  2. `python docs/generate_api_reference.py && git diff --exit-code docs/python_api/api_reference.md`
  3. `python docs/generate_validation_matrix.py --check`
  4. `python docs/check_doc_snippets.py`
  5. `python -m unittest tests.smoke_tests`

  If the Windows/bundled Python lacks `ruff`, use `Ubuntu-24.04` WSL with a
  temporary venv:
  `python3 -m venv /tmp/nmr-ci-venv && . /tmp/nmr-ci-venv/bin/activate && cd '/mnt/c/dev/MRSpinDynamics/PythonSpinDynamics' && python -m pip install -q -e '.[dev,opt,plot]'`.
  Then run the five smoke steps from that activated venv before committing or
  pushing.
- For PythonSpinDynamics updates in this repository, push completed and
  validated changes directly to `main` unless the user explicitly asks for a
  pull-request branch.
