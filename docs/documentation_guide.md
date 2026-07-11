# Documentation Guide

_Audited: 2026-07-11_

MRSpinDynamics contains user guides, generated references, validation records,
engineering design records, and future roadmaps. They serve different purposes;
reading an old design record as a current user guide is a common source of
confusion. Use this map to choose the right starting point.

## Start here

- New Python user: the [PythonSpinDynamics installation guide](../PythonSpinDynamics/docs/python_api/installation.md), then the [experiment workflow](../PythonSpinDynamics/docs/python_api/experiment_workflow.md).
- Learning by running code: the [organized examples catalog](../PythonSpinDynamics/docs/python_api/examples.md).
- Checking scientific support: the [validation matrix](../PythonSpinDynamics/docs/validation_matrix.md) and [known gaps](../PythonSpinDynamics/docs/python_api/known_gaps.md).
- Whole-workspace orientation: the [repository README](../README.md) and [roadmap](roadmap.md).
- Detailed reading or offline use: the [Python user manual](../PythonSpinDynamics/docs/user_manual.pdf).

## Document types

### User guides -- current behavior

Files under `PythonSpinDynamics/docs/python_api/` (except the generated API
inventory) explain supported concepts and workflows. The package README,
subproject READMEs, development-environment guide, and release process also
describe current behavior. Commands in these documents are expected to run.

### Generated references -- do not edit by hand

- `PythonSpinDynamics/docs/python_api/api_reference.md`
- `PythonSpinDynamics/docs/validation_matrix.md`
- `PythonSpinDynamics/docs/generated/validation_summary.tex`

Regenerate them with the scripts documented in the development guide.

### Engineering records -- implemented design history

Files whose names contain `plan`, `worklist`, or a milestone number often began
as implementation plans. Completed records now carry a dated status banner.
They remain useful for design rationale, equations, benchmarks, and known model
limits, but they are not the shortest route for learning the public API.

### Active roadmaps -- future work

`docs/roadmap.md`, `docs/publishing_plan.md`, and
`PythonSpinDynamics/docs/python_api/known_gaps.md` distinguish completed
foundations from remaining work. Future-looking claims should include a date or
an explicit status so they can be audited again.

### Validation and provenance records

Validation documents state what a test proves, its evidence level, and its
limitations. They should not be read as tutorials. Historical result logs are
retained for reproducibility even when newer summary matrices exist.

## Maintenance rule

When a feature lands, update its user guide, known-gap entry, roadmap status,
examples catalog, and manual in the same change. If a planning document is kept
for rationale, change its title or add a status banner so completed work is not
presented in the future tense.
