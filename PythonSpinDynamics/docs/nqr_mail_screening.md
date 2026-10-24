# NQR mail-screening system study

Use this guide to navigate the reproducible study that connects quadrupolar
spin dynamics to an absolute signal chain, spatial loading, complete acquisition
records and calibrated decisions. It lives in `studies/nqr_mail_screening`;
these are research scripts, not installed `spin_dynamics` entry points.
Start with [NQR models](python_api/nqr.md) for the underlying spin physics.

The numerical study has reached Phase 4. Its physical 1 g reference does **not**
demonstrate useful discrimination or an adaptive advantage. Numerical gates
refer to declared synthetic domains, not an installed screening instrument.

## Find the stage you need

| Stage | Implemented result | Source and reproduction instructions |
| --- | --- | --- |
| 0 | Frozen requirements, materials, evidence and test-set policy | [Phase 0](https://github.com/supertjhok/MRSpinDynamics/tree/main/PythonSpinDynamics/studies/nqr_mail_screening/phase0) |
| 1 | Absolute signal-chain reference, full spin pulse histories, motion and Zeeman spacing constraint | [Phase 1](https://github.com/supertjhok/MRSpinDynamics/tree/main/PythonSpinDynamics/studies/nqr_mail_screening/phase1) |
| 2 | Spatial transmit/receive maps, circuit/loading bounds and independent quasistatic material audit | [Phase 2](https://github.com/supertjhok/MRSpinDynamics/tree/main/PythonSpinDynamics/studies/nqr_mail_screening/phase2) |
| 3 | State-preserving pulse blocks, stop–transfer–stop timing, receiver/ADC/RFI and thermal records | [Phase 3](https://github.com/supertjhok/MRSpinDynamics/tree/main/PythonSpinDynamics/studies/nqr_mail_screening/phase3) |
| 4 | Calibrated fixed/adaptive decisions with correlated features and held-out evaluation | [Phase 4](https://github.com/supertjhok/MRSpinDynamics/tree/main/PythonSpinDynamics/studies/nqr_mail_screening/phase4) |

Phase 2's requirements file isolates its independent material-validation
dependencies. Generate its converged maps before reproducing the downstream
receiver and detector stages. Study caches and expensive raw artifacts live
under `.tmp`; checked-in reports record their provenance and gate scope.

## Interpret the detector results

At a 3.354 s parcel budget, the checked-in physical reference gives fixed-schedule
ROC AUC 0.509 and adaptive AUC 0.468. The paired interval does not establish an
adaptive benefit. Software-only amplitude multipliers of 10000 and 100000 are
diagnostic positive controls, not additional hardware gain or a sensitivity claim.
Untrained RFI drift raises false alarms above the illustrative 5% matched-domain
calibration setting. The detailed report retains uncertainty, target and receiver
temperature assumptions, time costs and train/calibration/test separation.

## Reproduce

From `PythonSpinDynamics`, using Ubuntu-24.04 for scientific verification:

```bash
python -m unittest tests.test_nqr_mail_screening_phase4
# Requires Phase 2 maps and the Phase 3 dependencies:
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase4/evaluate.py
python studies/nqr_mail_screening/phase4/plot_phase4.py
```

Full spatial and detector runs are not installation smoke checks. Read each
phase's README before running its gate. Physical signal strength, hardware
selection, blind material libraries and installed-site calibration remain open.
