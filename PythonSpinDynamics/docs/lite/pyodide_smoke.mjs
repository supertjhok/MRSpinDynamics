// Check that the package works in Pyodide, the in-browser Python that runs
// the JupyterLite notebooks on the documentation site.
//
//   cd docs/lite && npm ci && node pyodide_smoke.mjs <path to the package wheel>
//
// It installs the wheel, imports every spin_dynamics module, and runs each
// notebook source in notebooks/. An import that needs a standard-library
// module Pyodide lacks (multiprocessing, or sqlite3 without loading it) fails;
// optional third-party dependencies that Pyodide does not ship are reported.
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const wheel = process.argv[2];
if (!wheel || !fs.existsSync(wheel)) {
  console.error("usage: node pyodide_smoke.mjs <python_spin_dynamics-*.whl>");
  process.exit(2);
}

const pyodide = await loadPyodide();
console.log(`Pyodide ${pyodide.version}`);
// As in the notebooks: micropip resolves NumPy from the wheel's metadata and
// Matplotlib is installed by the first cell; SciPy is not loaded.
await pyodide.loadPackage(["micropip", "matplotlib"]);
const wheelName = path.basename(wheel);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheelName}`, fs.readFileSync(wheel));
pyodide.FS.mkdirTree("/notebooks");
const notebooks = fs.readdirSync(path.join(here, "notebooks")).filter((f) => f.endsWith(".py")).sort();
for (const name of notebooks) {
  pyodide.FS.writeFile(`/notebooks/${name}`, fs.readFileSync(path.join(here, "notebooks", name)));
}

const failures = await pyodide.runPythonAsync(`
import importlib, os, pkgutil, runpy, sys, time, traceback
import micropip

await micropip.install("emfs:/wheels/${wheelName}")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
import spin_dynamics

failures = []
optional = {}
modules = [m.name for m in pkgutil.walk_packages(spin_dynamics.__path__, "spin_dynamics.")]
for name in modules:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".")[0]
        if root in sys.stdlib_module_names or root == "spin_dynamics":
            failures.append(f"import {name}: {exc}")
        else:
            optional.setdefault(root, []).append(name)
    except Exception as exc:
        failures.append(f"import {name}: {type(exc).__name__}: {exc}")
print(f"imported {len(modules) - len(failures) - sum(map(len, optional.values()))} of {len(modules)} modules")
for root, names in sorted(optional.items()):
    print(f"  needs {root} (not in Pyodide): {', '.join(names)}")

for name in sorted(os.listdir("/notebooks")):
    start = time.perf_counter()
    try:
        runpy.run_path(f"/notebooks/{name}", run_name="__main__")
        print(f"ran {name} in {time.perf_counter() - start:.1f} s")
    except Exception:
        failures.append(f"notebook {name}:\\n{traceback.format_exc()}")
    plt.close("all")
failures
`);
const list = failures.toJs();
if (list.length) {
  console.error(`\n${list.length} failure(s):\n${list.join("\n")}`);
  process.exit(1);
}
console.log("Pyodide smoke test passed");
