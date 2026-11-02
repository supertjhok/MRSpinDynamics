# Try It in Your Browser

These notebooks run PythonSpinDynamics in your web browser with
[JupyterLite](https://jupyterlite.readthedocs.io/) and
[Pyodide](https://pyodide.org/), a build of Python for WebAssembly. Nothing is
installed on your computer and no server runs the code: the first cell
installs the package from a wheel published with this site, and NumPy and
Matplotlib come from Pyodide. The first run fetches about 35 MB (less on the
wire, as the WebAssembly runtime compresses well), which the browser then
caches. A notebook runs in a few seconds after that.

| Notebook | What it shows |
|---|---|
| [CPMG echo train](https://supertjhok.github.io/MRSpinDynamics/lite/notebooks/index.html?path=01_cpmg_echo_train.ipynb) | The experiment facade: plan and run a CPMG train, the transient before the asymptotic echo, and how the `"physics"` conventions change the decay. |
| [BPP relaxation](https://supertjhok.github.io/MRSpinDynamics/lite/notebooks/index.html?path=02_bpp_relaxation.ipynb) | `T1` and `T2` versus temperature for Arrhenius tumbling, and how the `T1` minimum moves with field. |
| [ESR powder spectrum](https://supertjhok.github.io/MRSpinDynamics/lite/notebooks/index.html?path=03_esr_powder_spectrum.ipynb) | Single-crystal lines and the absorption and derivative powder pattern of a rhombic `g` tensor. |
| [NQR powder nutation](https://supertjhok.github.io/MRSpinDynamics/lite/notebooks/index.html?path=04_nqr_powder_nutation.ipynb) | The spin-1 powder nutation curve and its `J_3/2` envelope, with the optimum near 119 degrees. |

[Open JupyterLab in the browser](https://supertjhok.github.io/MRSpinDynamics/lite/lab/index.html){ .md-button }

Edit any cell and press Shift+Enter to rerun it. Changes are kept in your
browser's storage, not on the site; download a notebook to keep it.

## Limits

- Pyodide runs in one thread with at most a few gigabytes of memory, so large
  3-D field solves and long optimizations belong on a desktop install.
  Worker options (`num_workers`) must stay at 1.
- Optional accelerators (Numba, JAX) and `h5py` are not available.
- `sqlite3` is loaded when a cell imports it directly; the package imports it
  only inside the NQR database reader.

## How the notebooks are built and tested

The notebook sources are Python files with `# %%` cell markers in
`docs/lite/notebooks/`. `docs/lite/build_lite.py` converts them to notebooks
and builds the JupyterLite site, which the documentation workflow publishes
under `/lite/`:

```bash
python -m pip install -e ".[lite]"
python docs/lite/build_lite.py --out site/lite
python -m http.server --directory site/lite
```

`tests/test_lite_notebooks.py` runs the sources under CPython, and the
`pyodide` CI job (`docs/lite/pyodide_smoke.mjs`) installs the wheel in the same
Pyodide version the site uses, imports every module, and runs each notebook.
To add a notebook, add a source file there; it is built and tested
automatically.
