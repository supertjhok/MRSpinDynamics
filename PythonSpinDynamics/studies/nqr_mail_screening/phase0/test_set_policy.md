# Phase 0 synthetic test-set policy — version 0.2.0

Frozen for numerical study under the requester's delegation to choose reasonable
defaults. This defines future sampler/evaluator contracts; no trials have yet run.

## Population and unit of evaluation

One trial is one complete parcel decision including all interrogated lines,
repetitions, retuning, recovery and processing. False alarms are counted per
parcel. Line-wise results are diagnostics, not independent parcel trials.
Report alarms, clears, indeterminate outcomes, exclusions and timeouts separately;
an indeterminate outcome must never be silently counted as a successful detection
or dropped from the denominator. The numerical outcome policy is specified below; operational routing is a later
deployment decision.

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

The reference quantitative H1 population is fentanyl HCl with equal allocation
over the 5 salt-mass and 3 crystalline-fraction values in study_defaults.md.
Material center is uniform in the admissible envelope volume, conditioned on the
whole inclusion fitting; orientation uses an isotropic powder model. Temperature
is uniform over 273.15–323.15 K. Geometry/loading distributions will be implemented
and checked in Phase 2. Exact boundary/temperature-endpoint cases are additional
stress strata, not evidence of real-world prevalence. Citrate is evaluated only
after its own spectroscopy is available and is never silently pooled as HCl.

H0 uses five equal 20% synthetic classes: paper/cardboard, polymer packing,
glycine, paracetamol and melamine in ordinary envelope packing. For the three
crystalline confusers, use the same mass and crystalline-fraction grid as H1.
Apply common geometry and temperature sampling to H0/H1 so a classifier cannot
win through different nuisance-generation rules. Empty aperture is a separate
instrument control, excluded from the headline mail ROC. Benign mixtures,
conductive contents and 50 mm mailers are separate stress tests until characterized.
Reweight H0 with each class emphasized at 60% (others 10% each), and report
per-class ROC. These are synthetic design priors, not measured prevalence.

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

## Statistical defaults

At each physical-time budget (3/5/10/30/60/120/300 s), score complete parcels with
a fixed-time protocol. AUC is computed from continuous scores with tied H1/H0
scores contributing one half. Threshold sweeps use the same records. Adaptive
stopping is a later comparison requiring scores calibrated at the declared
stopping rule; it must not obtain extra uncounted acquisition time.

Use 3000 H1 and 3000 H0 parent parcels in development, and a distinct 3000 + 3000
held-out synthetic set. Allocate H1 equally across the 15 mass/fraction strata
(200 each); allocate H0 equally across five classes (600 each). Use independent
partition seed namespaces: development 104729, held-out 130363, bootstrap 155921;
record the actual generator and stream mapping. A pilot of 200 per hypothesis
may debug the code, but cannot replace the nominal evaluation. Additional
edge/corner, endpoint and loading stress trials are reported separately.

Report 95% pointwise percentile intervals from 2000 parcel-level bootstrap
replicates stratified by the fixed H1/H0 strata. Reuse bootstrap indices for paired
design differences; repeated shots are not new parcels. Design ranking uses only
development data. Evaluate frozen finalists once on held-out data, without
selecting a new winner there. These intervals quantify finite synthetic sampling,
not model discrepancy or simultaneous coverage across many designs/time points.
Increase parcel counts/replicates if uncertainty is too large to distinguish
candidate improvements; no deployment PD/PFA guarantee is implied.

An absent/invalid score is a failed trial, not a dropped observation. Report
completion fraction and failures separately; headline AUC requires valid scores
for all planned parcels at that budget. Partial-case ROC is labeled diagnostic.
Any future operational policy routes indeterminate decisions to review, rather
than silently clearing them. Physical blind trial counts and confidence methods
will be set for the eventual deployment question in Phase 6.

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

## Freeze and changes

The requirements, library, defaults and policies are frozen in gate0_approval.json
with status `study_ready`, under delegated engineering judgment. Changes update
the requirement/library versions and normalized-text hashes after validation.
Provisional choices remain visibly provisional. Formal sign-off metadata remains
empty because this is not a hardware or experimental authorization.
