# Phase 0 completion audit — 2026-09-05

## What was missing

The four planned artifacts existed, but the readiness gate only counted 17
unresolved worksheet entries. It did not check material/test-set approval,
worksheet/reference-concept approval, document/schema conformance or an immutable
approval reference. Regex patterns in the schemas were over-escaped. Negative
hardware limits and malformed provisional values could pass semantic checks.
Population scope and validation policy lived only in prose rather than blocking
requirements. The README's escaped Markdown made its commands hard to use.

## Technical work completed

- Validate each worksheet/library/example against Draft 2020-12 JSON Schema,
  including dates, required fields, unknown properties and numeric bounds.
- Correct schema patterns; reject invalid provisional values, nonpositive
  physical limits and degenerate probability/confidence requirements.
- Protect mandatory requirement IDs, units, types and blocking flags.
- Add target-population, benign-population and test-policy requirements. The
  worksheet is now version 0.2.0 with 20 blocking entries, all initially unresolved.
- Supply a test-set policy covering parcel-level evaluation, data separation,
  replay, freeze/access rules, missing evidence and statistical decisions.
- Require stakeholder approval of the exact artifact set, detected with hashes;
  changed artifacts invalidate an existing approval. No approval is fabricated.
- Add regression checks for rejection paths as well as the valid draft.

## User inputs incorporated

See user_requirements.md. Requirements version 0.3.0 captures AUC/time objectives,
273.15–323.15 K, SLSE/SORC, pre-polarization comparison and no user-imposed RF cap.
Five previous scalar caps/operating-point fields are retained as optional nulls;
they are not blockers for the exploratory AUC/time question. Clearance and detailed
target scope are provisional. Exact envelope geometry remains unresolved.

## What still prevents Gate 0

The worksheet currently has 25 entries: 5 approved user choices, 2 provisional
interpretations and 18 unresolved entries (5 of those deliberately nonblocking).
Thus 15 blocking requirements still need resolution or approval. These cover
geometry/clearance, handling, amounts/populations, statistical policy, component
and thermal limits, and installation/test protocol. The entire artifact set also
needs stakeholder sign-off.

The two target candidates now match the pharmaceutical scope. Fentanyl HCl has
room-temperature direct-NQR evidence in Malone et al., Table 2:
https://doi.org/10.1093/pnasnexus/pgaf190 . Literature_supplement.json preserves the
reported uncertainties and keeps effective CPMG decay distinct from intrinsic T2.
Exact reference temperature and applicability across 0–50 C remain unresolved.

Fentanyl citrate has a published proton-T1-dispersion/DFT source:
https://doi.org/10.1016/j.ssnmr.2020.101697 . The inspected abstract supports a
candidate selection, not an imported direct-NQR pulse-response model. This spectroscopy gap must be resolved before using citrate in a quantitative
forward model; Phase 0 may approve an explicitly incomplete candidate library. Do not borrow HCl frequencies or
relaxation values for citrate. Salt-form candidates do not prove broad opioid
coverage. A second fully characterized target or additional source/calibration
work is still needed before running an executable two-target development set.

Benign records retain their local NQR database traceability. Missing linewidths,
relaxation, temperature references and uncertainty remain null; no values are
invented. Material-scale normalization, loading, RFI and pre-polarization gains
are subsequent-phase work. These literature observations are not scanner
performance claims for the requested envelope population.

## Completion boundary

The technical audit, user-scope revision, policy and validation repairs are
complete. **Phase 0 / Gate 0 is still open** for the listed requirement decisions,
approval. Second-target spectroscopy remains a later-model input gap. No model or scanner has been built.
