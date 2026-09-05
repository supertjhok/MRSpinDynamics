# Phase 0 completion audit — 2026-09-05

**Gate 0 is complete for the numerical design study.** The earlier 15-item blocker
list incorrectly treated routine engineering assumptions and later experimental
approvals as information the requester had to supply. The requester delegated
reasonable choices; requirements version 0.4.0 now records them explicitly.

## Completed

- Frozen the envelope and handling baseline: 420 x 330 x 25 mm loaded outer size,
  5 mm per-side clearance, stop-and-scan, 2 s initial handling/decision overhead.
- Defined time and amount sweeps, synthetic H1/H0 populations, 95% AUC confidence
  reporting, partitioning, counts, seeds and score-failure treatment.
- Chosen provisional electrical, receiver and thermal search limits with
  sensitivity ranges. No user-imposed RF power cap was introduced.
- Preserved SLSE/SORC and pre-polarization comparison requirements, with complete
  physical-time accounting and no assumed enhancement factor.
- Validated worksheet/library/example schemas, provenance references and source
  consistency; protected mandatory requirement identities, types and units.
- Frozen the study snapshot using platform-independent text hashes. Readiness
  accepts documented provisional defaults. Formal sign-off remains separate.

The worksheet has 25 entries: 5 explicit user choices, 13 documented provisional
choices, and 7 fields inapplicable to the current phase/objective. There are **zero
missing study requirements**. See study_defaults.md for values and rationale.

## Work belonging to subsequent phases

Fentanyl HCl has direct-NQR literature inputs in literature_supplement.json.
Fentanyl citrate is a selected candidate with incomplete pulsed-response evidence;
its spectroscopy cannot be borrowed from HCl. Proceed with the HCl Phase 1
reference; expand the material model when citrate evidence is resolved. Known
missing temperatures/relaxation/normalization remain unknown, not invented data.

Model-dependent loading, absolute transduction, current/voltage feasibility,
receiver recovery, thermal behavior and pre-polarization gain need implementation
and validation in Phases 1–3. Facility/material protocols and installation-specific
standards apply before physical work, not before a numerical requirement freeze.

`--require-ready` now passes with the frozen study defaults. This is not a claim
of scanner performance, certified ratings or permission to operate hardware.
