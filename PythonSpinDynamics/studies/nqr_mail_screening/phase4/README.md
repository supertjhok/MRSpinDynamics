# Phase 4: calibrated detection and adaptive acquisition

This is a numerical detector/protocol baseline built on the stopped-motion Phase 3
model. It is not a sensitivity claim for fentanyl screening or validation against
an installed mail-screening site. Read `phase4_report.json` for the gate result,
held-out uncertainty and supported scope.

## Spin histories and physical time

`tree.py` evaluates an actual three-step tree with three actions: SLSE-x, FID-y,
and SORC-x. All 39 prefixes are propagated at 0/20/50 C with the built-in full
three-level pulse Hamiltonian and affine relaxation machinery. Children inherit
the parent's complete powder/disorder density ensemble. A repeated action is a
new pulse train applied to that state, not reuse of a fresh-equilibrium waveform.
Coil position remains fixed and carrier-frame handoffs follow Phase 3.

The candidate retains the 1 s magnet dwell, 5 m/s transfer, 1.2 m spacing and
25 ms maximum coil dwell. One millisecond of recovery precedes subsequent blocks.
Both policies pay exactly the same entry, magnet dwell, transfer, exit, handling,
settling/calibration and final guard costs. This is 3.344 s before RF/recovery
costs. It is derived from the simulated complete cycle, with the maximum dwell
replaced by the duration actually used. The time sweep is 3.3475/3.351/3.354 s.
There is no uncharged re-preparation or independent-state averaging. This is a
small finite design domain; it does not establish an optimum over arbitrary times.

## Features, noise and likelihood

Two complex matched-filter projections per action retain early/late response,
providing echo-decay evidence as well as x/y-line information. The basis comes
from the nominal-temperature full spin response. Other temperatures are projected
into that same basis, without looking at held-out observations. A target phase
prior and 0.1/1 g amplitude prior are marginalized in the likelihood.

The covariance is fitted only to 1000 independent target-absent development
records passed through the Phase 3 causal receiver, ADC and three-reference
canceller. Receiver noise is fixed at 293.15 K, with nominal 7.3 ohm loaded
resistance and 1e-18 V2/Hz LNA noise; the 0/20/50 C variation is the target
spectral ensemble, not a full receiver-temperature uncertainty sweep. It includes correlations between early/late quadratures, between lines
and between acquisition times. A declared 5% identity shrinkage is applied after
standardization. Conditional Gaussian updates use the full cross-covariance with
previous observations; they are tested against a direct joint likelihood.

The larger Monte Carlo study uses this feature-level signal-plus-noise surrogate.
Switching-transient means are calculated from the same RF event/tail model and
included in target-absent calibration. Small-signal feature superposition is
checked against complete receiver/ADC processing with and without injection.
Reference-signal leakage and overload remain rejection/calibration problems from
Phase 3, not conditions this surrogate claims to repair. Uncertain switching
transients, carrier-specific site spectra and benign-material NQR libraries are
outside the validated feature domain.

## Adaptive decisions and calibration

`inference.py` uses the package's `ParticlePosterior`, `PredictiveModel`,
`AdaptiveDesignSession`, candidate space and Gaussian likelihood. The study utility
estimates reduction in binary classification Bayes risk, divided by the action's
physical seconds. It uses 48 common-random-number predictive samples per action
and retains Monte Carlo standard errors. The fixed comparator uses SLSE-x,
FID-y, then SORC-x, subject to the identical time budget.

The adaptive policy may stop when posterior risk is at most 0.05 **and** the Bayes
factor supplies at least 3:1 evidence. The evidence condition prevents stopping
solely because a skewed prior already favors absence. All final decision thresholds
are calibrated on 800 separate H0 records by a finite-sample order statistic.
The complete policy, including its adaptive selection and early stopping, is run
during calibration. The illustrative alpha=0.05 is a plotting/evaluation choice,
not a newly frozen operational false-alarm requirement.

The held-out evaluation uses 500 H0 and 500 H1 parcels with independent receiver
records and random target temperature/receive phase. Policies share the same
held-out parcels for paired comparisons, but training, calibration, testing and
shifted-RFI stress data use different seeds. Reports include ROC AUC with stratified
bootstrap intervals conditional on the fitted detector, exact binomial detection/false-alarm intervals, a conservative empirical
p-value-CDF check, actual physical times and paired AUC-difference intervals.

## Diagnostic and robustness cases

The physical reference is 1 g at the existing candidate's actual source voltage.
Separate **software-only amplitude multipliers of 10000 and 100000** probe
the intermediate-signal regime and provide a strong positive control
for the inference pipeline. It is not additional RF hardware gain, extra target
mass, or demonstrated achievable sensitivity. These results are always labelled
separately. Robustness cases use the intermediate multiplier to avoid hiding
sensitivity behind nearly perfect separation. They change the H1 prior to 1%, omit the 20 C library
record, remove y-line signal evidence, or introduce untrained RFI transfer drift.
Perfect empirical separation can give a degenerate bootstrap AUC interval;
this is not proof of perfect population performance. A shifted distribution may invalidate calibrated false-alarm performance; that is
reported rather than silently included in the matched-domain gate.

## Baseline findings

At the 3.354 s parcel budget, the physical 1 g reference gives fixed-schedule
AUC 0.509 (95% bootstrap interval 0.472–0.548) and adaptive AUC 0.468
(0.432–0.503). The paired adaptive-minus-fixed interval includes zero; this run
does not demonstrate useful discrimination or an adaptive advantage.

The intermediate software diagnostic gives AUC 0.675 fixed and 0.662 adaptive,
again without a demonstrated adaptive advantage. Untrained RFI drift raises
false alarms to 31.4% and 36.0%, respectively, despite the illustrative 5%
calibration setting. This deliberately fails the matched-distribution calibration
check. The numerical gate passes only for the declared matched synthetic domain.
The receiver feature-superposition check is within 0.006 fitted noise standard
deviations in its 78 cases. Phase 5 needs to address physical signal strength and
robustness; these results do not justify selecting this hardware candidate.

## Reproduce

From `PythonSpinDynamics` using the repository's Ubuntu-24.04 scientific runtime:

```bash
# Requires Phase 2's converged maps and Phase 3's dependencies.
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase4/evaluate.py
python studies/nqr_mail_screening/phase4/plot_phase4.py
python -m unittest tests.test_nqr_mail_screening_phase4
```

Raw tree, receiver-feature banks, fitted covariance and held-out scores are under
`.tmp/nqr_phase4`. Tree and receiver caches carry source/input fingerprints. The
checked-in report records model assumptions and numerical results. Phase 5 can
use this baseline to test better hardware and pulse candidates; installed-site calibration and blind material validation remain Phase 6.
