# Gate 1 closure and transition to Phase 2

**Passed for the declared absolute single-line reference.** This is not a measured
calibration or a detection-performance claim. The prior status incorrectly made
later-phase receiver/RFI, optimization and ROC work prerequisites for Phase 1.
The design plan assigns those to Phases 3–6.

`gate1_audit.py` runs the finite-pulse moving-sample model and closes a two-channel
I/Q integrate-and-dump reference receiver. Nonoverlapping valid exposure windows
have per-quadrature variance `S_one_sided / exposure_time`. The amplifier gain is
10,000 and each ADC has 16 bits across 2 V. Analog noise dithers the ADC. The
moment-to-flux-to-EMF path agrees with the voltage output, and whitened time-domain
and unitary-FFT SNR agree to numerical precision. This spectral check establishes
noise normalization; it does not independently validate the full spin solver.
The independent analytic ideal reference continues to check absolute spin scaling.

The seeded 2,048-trial ADC check reports 1.034% noise-SD discrepancy, 0.216%
discrepancy in the sqrt(four averages) law, and zero clipped samples. The actual
1 g signal is used; its very small SNR is calculated from the known template,
not estimated from the noisy sample mean. The 0.868-standard-error mean residual
is a consistency check, not an empirical low-SNR detection demonstration.

The optional preparation branch calls the built-in transport model and reports
its state enhancement, field and timing alongside the thermal control. Missing
fentanyl-specific microscopic inputs remain explicit assumptions. An illustrative
actuator budget is separated from RF energy: 0.1 kg moving mass, 50% drive
efficiency, 1 N drag, one acceleration, no regenerative credit; permanent-magnet
hold power is zero. These defaults do not select a transport mechanism.

Run from `PythonSpinDynamics`:

```bash
OPENBLAS_NUM_THREADS=1 python studies/nqr_mail_screening/phase1/gate1_audit.py
```

The audit exits nonzero on failure and records checks, tolerances, budgets and
source hashes in `gate1_report.json`. Phase 2 can now develop the spatial/aperture
model. The white reference receiver does not replace the tuned transfer, colored
noise, overload, switching or RFI implementation required in Phase 3.
