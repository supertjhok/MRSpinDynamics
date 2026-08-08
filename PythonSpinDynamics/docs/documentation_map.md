# How to Use This Documentation

PythonSpinDynamics covers several kinds of magnetic-resonance simulation. No
single reader needs all of them, and the shortest route is usually to begin
with an experimental question rather than a module name.

This page explains how the documentation fits together, what the recurring
terms mean, and where to go next. If you are new to the package, read this page
after [Installation](python_api/installation.md) and before opening the detailed
model references.

## Choose a route by what you are trying to do

| Goal | Start here | Continue with |
|---|---|---|
| Run a first NMR simulation | [Unified Experiment Workflow](python_api/experiment_workflow.md) | [Workflow catalog](python_api/workflows.md) and [examples](python_api/examples.md) |
| Adapt measurements to infer a sample property efficiently | [Bayesian experiment design](python_api/bayesian_design.md) | [Architecture and research plan](bayesian_experiment_design_plan.md) |
| Understand units and model choices | [Concepts and Units](python_api/concepts.md) | The relevant page under Physical Models |
| Import or construct a pulse sequence | [Sequence IR and Pulseq](python_api/sequence_ir.md) | [Phase Cycling](python_api/phase_cycling.md) |
| Simulate NQR, ESR, coupled spins, or exchange | The matching page under Physical Models | Its linked examples and validation records |
| Design a magnet, coil, or detector | The matching page under Fields and Hardware | Engineering, thermal, and validation pages |
| Model receiver arrays, SENSE, coupling, LNAs, or a birdcage | [Receiver-array imaging](receiver_array_imaging.md) | [SENSE](sense_imaging.md), [receiver networks](receiver_networks.md), and [birdcage coils](birdcage_coils.md) |
| Study a complete imaging system | [Portable Halbach MRI](portable_halbach_adaptive_mri.md) | [Q-space robustness](qspace_imaging_robustness.md) or [magnetic therapy](image_guided_magnetic_therapy.md) |
| Decide whether a result is trustworthy | [Validation guide](python_api/validation.md) | [Evidence matrix](validation_matrix.md) and [known gaps](python_api/known_gaps.md) |
| Look up a class or function | [Generated API reference](python_api/api_reference.md) | The narrative page for that model or workflow |

## The four layers used throughout the site

The documentation repeatedly separates four layers because each answers a
different question:

1. **Experiment description:** what sample, hardware, sequence, and acquisition
   are intended.
2. **Workflow:** the task-oriented function that assembles those inputs and
   returns a meaningful result, such as an echo train or reconstructed image.
3. **Model or numerical engine:** the equations and approximations used to
   propagate spins, particles, circuits, fields, or heat.
4. **Evidence:** the analytical limit, reference implementation, independent
   solver, published data, experiment, or regression check supporting a
   particular claim.

A workflow can be convenient without its model being appropriate for every
sample. A function can be well tested without having experimental validation.
The model and validation sections therefore matter whenever a result will be
interpreted quantitatively.

## Common terms

| Term | Meaning in this documentation |
|---|---|
| **Isochromat** | A subensemble of spins sharing the same offset, relaxation, and RF sensitivity. It is a simulation element, not a spatial voxel unless a workflow explicitly maps it to one. |
| **Field map** | A sampled spatial description of B0, B1, a gradient, or another field. A field solver creates it; an imaging or motion workflow consumes it. |
| **Particle distribution** | The spatial concentration or probability density of magnetic particles. In a simulation it may be represented by individual particle positions or by values on a grid. |
| **Operating state** | One realizable hardware configuration, such as a particular set of retained EPM remanences or coil currents. |
| **Facade** | The high-level `Experiment` interface that plans, validates, runs, and saves supported workflows. |
| **Direct workflow** | A public `run_*` or `simulate_*` function used when the experiment family and required inputs are already known. |
| **Reference model** | A deliberately small or dense implementation used for correctness and validation, not necessarily for large production calculations. |

## What kind of page are you reading?

- **User guide:** begins with a task and leads to a runnable result.
- **Model reference:** explains physical assumptions, inputs, outputs, and
  validity limits before presenting the API.
- **Design and validation note:** records why an implementation was chosen and
  how it was checked. These are useful for reviewers and developers, but they
  may be more detailed than a first-time user needs.
- **Generated reference:** lists API signatures or evidence records. Generated
  pages are for lookup and should be read alongside their narrative guide.
- **Internal architecture record:** excluded from the public navigation and
  retained to preserve design history. Its status block points to the current
  user-facing page.

## A useful reading order

For most work, the following sequence is enough:

1. read [Concepts and Units](python_api/concepts.md);
2. run one small example through the
   [Unified Experiment Workflow](python_api/experiment_workflow.md);
3. read the model page that matches the experiment;
4. reproduce the closest example; and
5. check the evidence matrix and known limitations before drawing a physical
   conclusion.
