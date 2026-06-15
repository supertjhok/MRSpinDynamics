# Examples

Examples live in `examples/`. They can be run from `PythonSpinDynamics` or from
inside `PythonSpinDynamics/examples`; each script adds the local `src` directory
to `sys.path` when the package has not been installed yet.

## Ideal CPMG

```powershell
python examples\ideal_cpmg.py --numpts 101
```

## Ideal FID

```powershell
python examples\ideal_fid.py --numpts 101
```

## Finite Ideal CPMG Train

```powershell
python examples\ideal_cpmg_train.py --numpts 101 --num-echoes 8
```

## Ideal Time-Varying CPMG

```powershell
python examples\ideal_time_varying_cpmg.py --numpts 101 --num-echoes 16
```

## Compare CPMG and FID

```powershell
python examples\compare_cpmg_fid.py --numpts 101
```

Optionally save arrays:

```powershell
python examples\compare_cpmg_fid.py --numpts 101 --save-npz results\ideal_compare.npz
```

## Export Validation Arrays

```powershell
python examples\export_validation_arrays.py results\validation_arrays.npz --numpts 101
```

## Plot Ideal Workflows

This example requires Matplotlib.

```powershell
python examples\plot_ideal_workflows.py --numpts 201 --output results\ideal_workflows.png
```

The plotting example narrows the FID offset range by default so the FID panel is
readable. To reproduce the raw MATLAB default FID range, use:

```powershell
python examples\plot_ideal_workflows.py --fid-maxoffs 10 --raw-fid-scale
```

## Plot Ideal Imaging

This example uses the `flower.png` phantom from the MATLAB reference tree. It
uses Matplotlib when available and falls back to Pillow for writing the output
PNG.

```powershell
python examples\plot_ideal_imaging.py --pixels 6 --ny 7 --output results\ideal_imaging.png
```

## Plot Custom Imaging Fields

This example builds a small synthetic phantom with custom B0, transmit-B1, and
receive-B1 maps, then plots the input maps, k-space, and reconstruction. It
requires Matplotlib.

```powershell
python examples\plot_custom_imaging_fields.py --pixels 8 --ny 7 --output results\custom_imaging_fields.png
```

## Tuned-Probe CPMG

```powershell
python examples\tuned_probe_cpmg.py --numpts 101
```

## Probe CPMG Comparison

```powershell
python examples\probe_cpmg_compare.py --numpts 101
```

## Probe Parameter Sweeps

```powershell
python examples\probe_parameter_sweeps.py --numpts 101
```

## Matched CPMG-IR Finite Train

```powershell
python examples\matched_cpmg_ir_train.py --numpts 21 --num-echoes 4 --num-tau 4
```

## Finite Probe Train Sweeps

```powershell
python examples\finite_probe_train_sweeps.py --numpts 21 --num-echoes 3
```

## Matched Diffusion CPMG

```powershell
python examples\matched_diffusion_cpmg.py --numpts 21 --num-echoes 3
```

## Plot Probe Parameter Sweep

This example requires Matplotlib.

```powershell
python examples\plot_probe_parameter_sweep.py --probe tuned --sweep q --output results\tuned_q_sweep.png
```

## Plot Probe CPMG Comparison

This example requires Matplotlib.

```powershell
python examples\plot_probe_cpmg.py --numpts 101 --output results\probe_cpmg.png
```

The asymptotic magnetization panel shows magnitude by default because tuned and
untuned probe responses can rotate most of the signal into the imaginary
component. Use `--masy-component real`, `imag`, or `phase` to inspect a specific
component.

## Plot Optimization Workflows

This example requires Matplotlib. It runs compact tuned-probe refocusing and
target-excitation optimization diagnostics, then probes the current
inverse-excitation objective. The inverse stage starts from the phase-flipped
target pulse, then uses a MATLAB-style multi-start strategy that perturbs the
best inverse found so far. Treat the inverse panel as a parity diagnostic, not
as a validated inverse pulse-design recipe yet.

```powershell
python examples\plot_optimization_workflows.py --numpts 11 --segments 2 --starts 2 --inverse-starts 4 --output results\optimization_workflows.png
```

The defaults are intentionally small. Increase `--numpts`, `--segments`,
`--starts`, `--excitation-segments`, `--inverse-starts`, and the optimizer
settings when using it as the start of a real pulse-design study.

## Diagnose Optimization Backends

This non-plotting diagnostic compares the NumPy pattern-search fallback, the
optional SciPy backend, and a random inverse-candidate baseline. It prints
objective values and residual/target area ratios so optimizer behavior can be
debugged without relying on plots.

```powershell
python examples\diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

On WSL2, create a virtual environment and install the optional SciPy backend
before running the comparison:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[opt]"
python examples/diagnose_optimization_backends.py --backend all --numpts 21 --segments 3
```

## Plot Finite Train Workflows

This example requires Matplotlib. It compares finite CPMG echo trains across
ideal, tuned, untuned, and matched probe models.

```powershell
python examples\plot_finite_train_workflows.py --numpts 17 --num-echoes 4 --output results\finite_trains.png
```

## Plot Diffusion Sweep

This example requires Matplotlib. It plots a compact matched-probe diffusion
CPMG Q sweep, including echo-integral decay and a Q-by-echo heatmap.

```powershell
python examples\plot_diffusion_sweep.py --numpts 17 --num-echoes 3 --output results\diffusion_sweep.png
```

## Plot Time-Varying Field Sweep

This example requires Matplotlib. It visualizes the ideal time-varying-field
CPMG amplitude sweep, including the B0 waveform and final echoes.

```powershell
python examples\plot_time_varying_sweep.py --numpts 51 --num-echoes 12 --output results\time_varying_sweep.png
```
