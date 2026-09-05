# Phase 0 test-set and evidence policy (draft, version 0.1.0)

This is a reviewable study policy, pending stakeholder approval. It defines
contracts for later samplers and trials; it does not create experimental data.

## Population and unit of evaluation

One trial is one complete parcel decision including all interrogated lines,
repetitions, retuning, recovery and processing. False alarms are counted per
parcel. Line-wise results are diagnostics, not independent parcel trials.
Report alarms, clears, indeterminate outcomes, exclusions and timeouts separately;
an indeterminate outcome must never be silently counted as a successful detection
or dropped from the denominator. The stakeholder must approve its operational
routing and metric treatment before performance evaluation.

The development target candidates are fentanyl hydrochloride and fentanyl citrate.
The HCl entry includes directly observed literature 14N lines; the citrate entry
has only a source lead for inferred NQR parameters and no imported pulsed-response
record. Glycine, paracetamol and melamine remain benign crystalline candidates.
Null controls cover empty aperture, paper/cardboard and polymer packing. Add
representative lawful pharmaceutical forms and excipients when the benign
population is frozen. Chemical identity alone cannot determine legal status or
prescription context; the instrument can flag material evidence for review.
Null means target-absent, not zero loading or interference. The empty aperture
is an instrument control, not a representative benign-mail population.

The target-scope requirement must define amount by material, crystalline fraction,
material form, mixtures, parcel geometry, location and pose. The benign-scope
requirement must define contents and sampling weights, conductive/dielectric
loading, environment and site RFI. Include difficult benign cases and declared
out-of-scope contents; report each stratum, not just a pooled average. Unknown
weights prevent operational aggregate PD/PFA claims.

## Separation and replay

Assign each underlying parcel, specimen/lot and acquisition session to exactly
one partition before generating augmented or repeated records:

| Partition | Permitted use | Freeze rule |
| --- | --- | --- |
| development | Physics debugging, fitting, design ranking and threshold selection | Version data and record all changes |
| held_out_synthetic | Evaluation of a frozen model, design and threshold | Independent scenario IDs and random streams; no tuning on these results |
| blind_experimental | Final independent empirical evaluation | Custodian holds truth labels until predictions are sealed |

Use identical development scenarios and noise streams to compare candidates.
Seeds alone do not ensure independence: record parent parcel/session IDs,
generator version, partition, configuration hash and raw-data hashes. Split at
the parent level so replicas cannot leak across partitions. A held-out result
used to change a model, threshold or design becomes development data; obtain a
new untouched set for the next final evaluation.

Freeze the requirements/library versions, model commit, scenario configuration,
decision rule and threshold before accessing held-out outcomes. Preserve the
freeze reference and custodian/access policy alongside every held-out record.
The future scenario sampler must enforce these contracts; none exists in Phase 0.

## Statistics to resolve before Gate 0

The approved test policy must name the AUC uncertainty method and, where used,
the PD lower-bound and PFA upper-bound method,
confidence level, whether confidence is simultaneous across strata, and how
multiplicity is handled. Choose trial counts from the approved bounds and expected
outcome precision; no universal trial count is supplied here. Distinguish finite
Monte Carlo uncertainty, model discrepancy, and experimental uncertainty.
Do not infer zero population false-alarm probability from zero observed alarms.

## Evidence and missing values

Every input and derived claim uses evidence_classes.json. A result-level label
does not replace component-level labels. Preserve source record IDs and input
versions, units, uncertainty model (or an explicit unknown), and transformation
history. Stakeholder limits carry an owner and approval reference separately.

Do not replace missing relaxation times, reference temperatures, linewidths,
line strengths, density, isotope/site counts or crystalline fractions with zero.
An exploratory imputation must be a separately labeled predicted prior with
justified bounds and sensitivity analysis. It cannot overwrite literature data.
Do not treat T2, T2-star and echo-train decay as interchangeable observables.

The candidate library is sufficient to select development materials for Phase 0;
it is not sufficient for an absolute signal or screening-performance prediction.
Phase 1 must resolve normalization and applicable calibration for its selected
reference line. Later phases resolve loading, interference and blind trials.

## Approval and changes

Resolve all worksheet entries, approve the reference concept, and freeze the
material library. Record stakeholder identity, UTC date and decision reference
in gate0_approval.json, with SHA-256 of every listed artifact after final edits.
Any change to those bytes invalidates the recorded approval and requires review
of the new revision. A digest is a change detector, not an identity signature.
