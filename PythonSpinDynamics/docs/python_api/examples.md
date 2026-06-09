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
