# MATLABSpinDynamics

MATLABSpinDynamics is the MATLAB NMR reference implementation within the
MRSpinDynamics repository. It contains the original MATLAB functions, successive
Version 2 and Version 3 workflows, probe-circuit models, pulse-design helpers, imaging
examples, and validation material used by the Python port.

The active implementation is in:

```matlab
Version_3/code
```

The older `Version_1` and `Version_2` trees are kept for
historical comparison.

## Physical Scope

The simulator assumes a bath of uncoupled spin-1/2 nuclei in a possibly
non-uniform and time-varying `B0` field. The sample is represented as a
collection of isochromats, and each isochromat evolves as a classical
magnetization vector under RF rotations, free precession, optional relaxation,
and optional probe-circuit response.

The code does not explicitly model spin-spin coupling, spins with `I > 1/2`,
multi-quantum pathways, density-matrix entanglement, or quadrupolar dynamics.
Those effects only enter indirectly if they are folded into effective
relaxation constants or supplied field maps.

## Installation Requirements

No MATLAB toolbox package installation is required. Clone or copy the repository
and add the active Version 3 code tree to the MATLAB path:

```matlab
repo = pwd;  % run from MATLABSpinDynamics
addpath(genpath(fullfile(repo, 'Version_3', 'code')));
```

Use a recent MATLAB release. The exact minimum release has not been pinned; the
current code uses ordinary MATLAB scripts/functions plus features such as
`parfor`, local helper functions, and modern graphics/export workflows.

Core ideal CPMG/FID and many basic probe examples use base MATLAB. Additional
workflows require optional components:

- Parallel Computing Toolbox: scripts that use `parfor`, including many
  sweeps, imaging workflows, CPMG-IR studies, and optimization repeats.
- Optimization Toolbox: matched-probe network design and `opt_pulse` workflows
  that call `fmincon` or `optimoptions`.
- Image Processing Toolbox: imaging examples that call `imresize` or `rgb2gray`.
- MATLAB Coder plus a configured C/C++ compiler: optional MEX/code-generation
  build scripts under `code/mex`. Generated MEX/codegen outputs are local
  build artifacts and are ignored by Git.
- `export_fig`: optional third-party File Exchange helper used by some plotting
  or export scripts through `safe_export_fig`. When unavailable, exports are
  skipped with a warning.

Literature PDFs and generated MATLAB artifacts such as `.fig`, `.mat`, MEX
outputs, codegen reports, and result folders are intentionally ignored. The
self-authored reports under `Version_1/reports` remain part of the source
tree.

## Documentation

- `docs/user_manual.tex` and `docs/user_manual.pdf` are the main user manual,
  with assumptions, equations, cartoons, installation requirements, workflow
  notes, and an API-style function-family reference.
- `docs/QUICK_START.md` gives a practical MATLAB entry point.
- `docs/VERSION_GUIDE.md` maps active and legacy folders.
- `docs/VERSION_3_WORKFLOWS.md` summarizes Version 3 workflow scripts.
- `docs/SPEED_AUDIT.md` records performance notes and acceleration targets.

Build the manual from this directory with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs\user_manual.tex
```

## Quick Start

```matlab
repo = pwd;  % run from MATLABSpinDynamics
addpath(genpath(fullfile(repo, 'Version_3', 'code')));

numpts = 2001;
maxoffs = 20;
del_w = linspace(-maxoffs, maxoffs, numpts);

tacq = 5*pi;
window = sinc(del_w*tacq/(2*pi));
window = window ./ sum(window);

texc = [pi/2 -1];
pexc = [pi/2 0];
aexc = [1 0];

tref = pi*[3 1 3];
pref = [0 0 0];
aref = [0 1 0];

neff = calc_rot_axis_arba3(tref, pref, aref, del_w, 0);
masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq);
masy2 = sim_spin_dynamics_asymp_mag3(texc, pexc + pi, aexc, neff, del_w, tacq);
masy = (masy1 - masy2)/2;
[echo, tvect] = calc_time_domain_echo(masy, del_w, 0, 0);
```

For more complete runnable workflows, start with `docs/QUICK_START.md` and the
scripts in `Version_3/code`.

## Python Port

The sibling `../PythonSpinDynamics` repository contains the Python port. Its
README and manual describe Python-specific installation, validation status,
examples, and API conventions. The MATLAB Version 3 implementation remains the
numerical reference for the port.

## Contact

Questions can be sent to Soumyajit Mandal (`supertjhok@gmail.com`).
