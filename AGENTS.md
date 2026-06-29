# Project Memory

- For PythonSpinDynamics examples or tests that need SciPy, Matplotlib, or a Linux
  plotting environment, use WSL distribution `Ubuntu-24.04`. It already has
  `numpy`, `scipy`, and `matplotlib` available. Example:
  `wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/Users/super/OneDrive/Codex/NMR/PythonSpinDynamics' && python3 examples/plot_dexsy_exchange.py --output .tmp/dexsy_exchange_wsl.png"`.
- Avoid using `Ubuntu2404Codex` for this repository's plotting/NNLS verification
  unless explicitly requested; it did not have the needed Python packages when
  last checked.
- Before pushing PythonSpinDynamics updates, reproduce the GitHub smoke job, not
  just `python -m unittest tests.smoke_tests`. The smoke workflow also runs
  `python -m ruff check src tests examples` and
  `python docs/generate_api_reference.py && git diff --exit-code docs/python_api/api_reference.md`.
  If the Windows/bundled Python lacks `ruff`, use `Ubuntu-24.04` WSL with a
  temporary venv:
  `python3 -m venv /tmp/nmr-ci-venv && . /tmp/nmr-ci-venv/bin/activate && cd '/mnt/c/Users/super/OneDrive/Codex/NMR/PythonSpinDynamics' && python -m pip install -q -e '.[dev,opt,plot]'`.
  Then run the three smoke steps from that activated venv before committing or
  pushing.
- For PythonSpinDynamics updates in this repository, push completed and
  validated changes directly to `main` unless the user explicitly asks for a
  pull-request branch.
