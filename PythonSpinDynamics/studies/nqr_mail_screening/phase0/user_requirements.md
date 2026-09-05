# User requirements — 2026-09-05

Source: the study requester in the current Codex task. This is requirement origin,
not scientific evidence. Exact explicit choices are recorded as approved inputs;
the complete worksheet and Gate 0 remain unapproved.

- Screen typical USPS/UPS/FedEx envelopes; allow approximately 5 mm clearance.
  Per-side clearance is a provisional interpretation, not an approved aperture.
- Minimize time per envelope and maximize ROC AUC. Retain a Pareto frontier;
  evaluate AUC at matched full physical times. No fixed scan-time cap supplied.
- Target illicit pharmaceuticals, with synthetic opioids prioritized. Fentanyl
  HCl and fentanyl citrate are proposed distinct material forms; they do not
  establish coverage across different opioid molecules or all illicit drugs.
- Operate over 0–50 degrees C (273.15–323.15 K).
- No user-imposed RF power cap. Engineering current, voltage, thermal and
  installation limits still constrain achievable power.
- Use pulsed SLSE/SORC measurements and compare pre-polarization against the
  unpolarized baseline. No gain or polarizing-field strength is assumed.

## Consequences for later phases

ROC AUC measures discrimination over thresholds; it does not specify a deployable
alarm threshold. Fixed PD/PFA constraints are optional during this exploratory
study. Freeze target/benign distributions and AUC confidence methodology before
ranking designs. Compare AUC(time) and report uncertainty and class-specific ROC
curves; full AUC can hide behavior at the low-false-alarm end. A low-PFA partial
AUC is a useful diagnostic once the range is defined, not a new user requirement.

Pre-polarization must be a separately costed hardware/protocol branch. Include
polarization buildup, field ramp/transfer, residual-field settling, relaxation,
extra heating and cycle overhead. Compare at equal total envelope time, not equal
number of echoes. Model or measure material-specific polarization transfer; do
not multiply the 14N equilibrium response by an assumed enhancement factor.

Published mail-package NQR work discusses proton-to-14N polarization transfer as
an SNR-improvement approach; this motivates evaluating the branch, not assigning
a gain for fentanyl salts:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10073136/

Remaining scope details: loaded outer envelope dimensions/thickness, clearance
convention, exact target forms and amounts, benign prevalence/mixtures, confidence
method, stop/indexed handling, hardware/thermal limits and installation protocol.
