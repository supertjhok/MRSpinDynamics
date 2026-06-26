# Phase Cycling Findings

Yes. The Python package had useful phase-related machinery, but phase cycling
was not handled consistently as a first-class concept. An initial
`spin_dynamics.phase_cycling` layer now owns reusable scan-table rows, receiver
weights, and the default two-step CPMG/PAP branch combination. Current support
is still partial: absolute RF phase tracking, receiver/display conventions, and
some pathway approximations remain distributed across workflow-specific code.

## What Exists

- `spin_dynamics.phase_cycling` provides `PhaseStep`, `PhaseCycle`, and
  `cpmg_two_step_phase_cycle`. Rows are scan steps in the cycle. Columns are
  named logical RF pulse roles/events, not unique phase values. A row also
  carries `receiver_phase_rad`, `weight`, and an optional label. Branch
  combination uses `weight * exp(-1j * receiver_phase_rad)` and, by default,
  normalizes by the sum of absolute branch weights.
- Finite CPMG workflows now use the default `PhaseCycle` for their two-step
  excitation cycle. The default table has one pulse column, `excitation`, two
  rows with phases `pi/2` and `3*pi/2`, and branch weights `+1` and `-1`, so it
  reproduces the previous `(branch1 - branch2) / 2` result. See
  `src/spin_dynamics/workflows/cpmg.py`.
- PGSTE walker results now expose `pgste_stimulated_echo_phase_cycle()` as
  `result.phase_cycle`. This is a one-row selected-pathway table with pulse
  columns `excitation_90`, `store_90`, and `read_90`. The PGSTE signal is not a
  receiver-weighted sum over explicit simulated branches; pathway selection is
  modeled by the spoiler and `mth=0.0` equilibrium-suppression convention.
- `diff_stebp_phase_cycle()` is a full 16-step table reproducing the Bruker
  `diff_stebp.gp` 13-interval bipolar PGSTE phase program: `ph1` on
  `excitation_90`, `ph2` on `store_90`, `ph3` on `read_90`, `ph4` on both
  `refocus_180` pulses, and the receiver `ph31`. The receiver satisfies
  `ph31 = -(ph1 + ph2 + ph3) mod 4`, selecting the stimulated-echo pathway
  `Delta p = (+1, -2, +1, +1, -2)`. The bipolar 13-interval diffusion workflow
  (`spin_dynamics.workflows.bipolar`) exposes it as `result.phase_cycle`. See
  [Bipolar 13-Interval PGSTE](bipolar_pgste.md).
- The original/reference ideal, tuned, untuned, and matched asymptotic CPMG
  paths also route their default two-step subtraction through
  `cpmg_two_step_phase_cycle`.
- `spin_dynamics.absolute_phase` tracks laboratory-frame pulse phases,
  optional phase binning, and per-pulse matrix reuse for finite CPMG-like
  trains. Its `FiniteCPMGPulsePlan` is described as phase cycling, but it only
  encodes the CPMG two-branch excitation pattern plus optional per-refocusing
  absolute-phase matrices. It is not a general scan table.
- Probe-aware finite CPMG workflows can use the absolute phase schedule to
  solve phase-dependent tuned, untuned, or matched pulse shapes. This is
  valuable for absolute-phase transient studies, but it is separate from a
  conventional phase-cycle table with transmitter phases, receiver phases, and
  branch weights.
- PGSTE documentation and code model a selected stimulated-echo pathway by
  setting `mth=0.0` during the walker sequence. That suppresses equilibrium
  regrowth into an unwanted FID and is now recorded as a one-row `PhaseCycle`,
  but it is not yet a multi-branch explicit phase-cycling simulation.
- NQR and ESR pulsed helpers expose individual RF phases for FID and echo
  simulations, but they do not expose reusable phase programs, receiver phase
  cycling, or pathway selection tables.

## Inconsistencies

- The word "phase" currently means different things in different places:
  rotating-frame RF phase, absolute laboratory RF phase, imaging phase encode,
  phase-bin cache key, receiver phase metadata, and selected coherence pathway.
  Those are related but not interchangeable.
- There is now a public `PhaseCycle` object, but public workflows do not yet
  accept arbitrary user-supplied phase-cycle tables.
- Receiver phase is explicit in `PhaseStep` branch combination, but most
  workflow APIs still expose only the default CPMG/PAP table rather than a
  user-configurable receiver-phase program.
- The implemented CPMG cycle is now owned by a common phase-cycle executor in
  `workflows.cpmg`, but similar two-branch patterns in imaging, diffusion,
  time-varying-field, WURST, optimization, and IR workflows remain to be
  factored.
- Absolute-phase binning solves a performance and transient-model problem, but
  the API shape can look like phase cycling. A user could reasonably expect
  `phase_bins` to mean cycling or averaging over bins, while it actually means
  "snap scheduled absolute pulse phases to reusable pulse matrices."
- Pathway selection is only partly explicit. CPMG branch subtraction and PGSTE
  selected-pathway metadata are documented under `PhaseCycle`, but
  gradient/spoiler assumptions still select or suppress unwanted pathways in
  workflow-specific code.
- Tests now include workflow-independent phase-table fixtures: a four-step
  CYCLOPS-style receiver cycle and a three-pulse stimulated-echo pathway
  selector. These validate receiver-phase filtering separately from any one
  workflow's pulse or motion implementation.

## Consequences

- Extending beyond the built-in CPMG two-step cycle will require copying and
  modifying workflow internals.
- Adding receiver cycling, EXORCYCLE-style schemes, CYCLOPS-like receiver
  correction, stimulated-echo tables, coherence-order filters, or ESR/NQR phase
  programs would likely produce another one-off implementation unless the API
  is factored first.
- Current documentation can make absolute phase look more complete than it is:
  absolute-phase tracking is a real feature, but it is not a general phase
  cycling system.

## Recommended Direction

- Build out the initial public phase-cycle model already added in
  `spin_dynamics.phase_cycling`:
  `PhaseStep(pulse_phases, receiver_phase_rad=0, weight=1, label=None)` and
  `PhaseCycle(steps, pulse_names=None, normalize=True)`.
- Keep absolute-phase tracking separate. A phase-cycle step should define the
  requested rotating-frame phases and receiver phase for a scan branch; the
  existing `AbsolutePhaseSpec` should then compute laboratory-frame pulse
  phases from timing.
- Provide named constructors for the existing built-ins, starting with the
  current two-step CPMG/PAP pattern and the selected PGSTE stimulated-echo
  pathway. Then finite CPMG workflows can call a shared executor and preserve
  current default results.
- Make receiver phase explicit in signal combination:
  `combined = sum(weight * exp(-1j * receiver_phase) * branch_signal)`.
- Store phase-cycle metadata on results next to `absolute_phase`, including the
  table, branch labels, pulse columns, receiver phases, and normalized weights.
- Keep workflow-independent tests for phase-cycle combination alongside
  workflow tests showing that the named CPMG/PAP and PGSTE constructors
  reproduce current default behavior.
- Document which workflows only support fixed built-in cycles and which accept
  arbitrary cycle tables as that support grows.

## Suggested Scope

The first implementation is intentionally conservative: the existing CPMG
two-step subtraction is factored into a reusable model without changing
numerical defaults, and PGSTE records the selected stimulated-echo pathway as
metadata. Next, add arbitrary user-supplied cycle-table support to finite CPMG
train workflows. Only then should the same abstraction be extended to NQR and
ESR, where pathway selection and receiver conventions need more domain-specific
care.
