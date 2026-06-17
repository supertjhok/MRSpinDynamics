# Quick Start

This guide gets a new MATLAB user to a working `Version_2` simulation quickly.
The active MATLAB code lives under:

```text
SpinDynamicsUpdated/Version_2/code
```

The older `SpinDynamics` and `SpinDynamicsUpdated/Version_1` folders are useful
for historical reference, but new work should start from `Version_2`.

## 1. Open MATLAB at the repository root

Start MATLAB with the current folder set to the repository root, the folder that
contains `README.md`, `SpinDynamics`, and `SpinDynamicsUpdated`.

On this machine that path is:

```text
C:\Users\super\OneDrive - Brookhaven National Laboratory\Codex\NMR\MATLABSpinDynamics
```

## 2. Add the active code tree to the MATLAB path

Run this from the repository root:

```matlab
addpath(genpath(fullfile(pwd,'SpinDynamicsUpdated','Version_2','code')));
```

This adds the active parameter constructors, examples, simulation routines,
probe circuit models, image assets, and helper functions.

## 3. Run the simplest CPMG smoke test

The ideal-probe CPMG asymptotic example is the best first check because it does
not need image assets, parallel workers, MEX files, or external result files.

```matlab
run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'CPMG_Asymp_Examples','noProbeEffects_CPMG_Asymp.m'));
```

Expected result:

- a figure showing real and imaginary asymptotic magnetization versus normalized
  offset frequency;
- a figure showing the real and imaginary time-domain echo;
- workspace variables including `sp`, `pp`, `masy`, `echo_asy`, and `tvect`.

Key functions used:

- `set_params_ideal`
- `calc_masy_ideal`
- `calc_rot_axis_arba3`
- `sim_spin_dynamics_asymp_mag3`
- `calc_time_domain_echo`

## 4. Try the probe-effect examples

The comparable CPMG examples are:

```matlab
run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'CPMG_Asymp_Examples','UntunedProbeEffects_CPMG_Asymp.m'));

run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'CPMG_Asymp_Examples','TunedProbeEffects_CPMG_Asymp.m'));

run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'CPMG_Asymp_Examples','MatchedProbeEffects_CPMG_Asymp.m'));
```

These examples compare how untuned, tuned, and matched probe models alter the
asymptotic magnetization, received spectrum, echo, and SNR.

## 5. Try the FID example

```matlab
run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'FID_Example','noProbeEffects_FID.m'));
```

Expected result:

- workspace variables including `sp`, `pp`, and `mrx`.

Key functions used:

- `set_params_ideal_FID`
- `simFID_ideal`

## 6. Try an imaging example

The imaging examples use `flower.png`, which is stored in:

```text
SpinDynamicsUpdated/Version_2/code/Images/flower.png
```

Because the full `Version_2/code` tree was added to the path in step 2, MATLAB
should be able to find this image.

The imaging simulation drivers use `parfor` internally for speed, so they
require the Parallel Computing Toolbox as written. To run them without that
toolbox, edit the `parfor` loops in the corresponding `Sim_CPMG` image
simulation function to ordinary `for` loops; the calculation will be slower.

```matlab
run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'Imaging_demo','imaging_example_ideal.m'));
```

Expected result:

- workspace variables including `params` and `echo_int_all`.

The tuned and matched imaging examples are also available:

```matlab
run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'Imaging_demo','Imaging_example_tuned.m'));

run(fullfile('SpinDynamicsUpdated','Version_2','code', ...
    'Imaging_demo','imaging_example_matched.m'));
```

## Toolboxes and Dependencies

Most basic examples use standard MATLAB functionality. Some workflows need
additional tools:

- Parallel Computing Toolbox: scripts/functions using `parfor`, including the
  imaging simulation drivers under `code/Sim_CPMG`; imaging demos also include
  optional commented `parpool` calls.
- MATLAB Coder: MEX build scripts under `code/mex`.
- Image Processing Toolbox: imaging examples use `imresize` and `rgb2gray`.
- `export_fig`: some plotting/export scripts call `export_fig` with hard-coded
  output paths.

## Troubleshooting

If MATLAB cannot find a function, rerun:

```matlab
addpath(genpath(fullfile(pwd,'SpinDynamicsUpdated','Version_2','code')));
```

If MATLAB cannot find `flower.png`, check that this path is on the MATLAB path:

```text
SpinDynamicsUpdated/Version_2/code/Images
```

If a script exports to a missing Dropbox or Overleaf path, edit the `export_fig`
line or comment it out before running.

If a script uses `parfor` and the Parallel Computing Toolbox is unavailable,
convert the loop to `for` for a slower serial run.
