# Project Memory

- For PythonSpinDynamics examples or tests that need SciPy, Matplotlib, or a Linux
  plotting environment, use WSL distribution `Ubuntu-24.04`. It already has
  `numpy`, `scipy`, and `matplotlib` available. Example:
  `wsl.exe -d Ubuntu-24.04 -- bash -lc "cd '/mnt/c/Users/super/OneDrive - Brookhaven National Laboratory/Codex/NMR/PythonSpinDynamics' && python3 examples/plot_dexsy_exchange.py --output .tmp/dexsy_exchange_wsl.png"`.
- Avoid using `Ubuntu2404Codex` for this repository's plotting/NNLS verification
  unless explicitly requested; it did not have the needed Python packages when
  last checked.
