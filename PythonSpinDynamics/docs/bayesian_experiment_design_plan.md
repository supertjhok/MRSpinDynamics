# Bayesian experiment design and adaptive acquisition

> **Status (2026-07-16): Phases 0--3 implemented.** An exact-grid
> inversion-recovery reference, a generic NumPy grid/particle design core, and
> planner-validated facade adapters for CPMG-IR, deterministic PGSE, NQR FID,
> and ESR Hahn echo, plus validated acceleration and robustness tools, are
> operational. This is not a claim that every
> PythonSpinDynamics workflow can already be planned adaptively.

Magnetic-resonance experiments often use a fixed, uniformly or logarithmically
spaced acquisition schedule even when early measurements already rule out much
of the plausible sample behavior. Bayesian experimental design instead treats
the next acquisition as a decision under uncertainty: predict what each
feasible acquisition might measure, estimate how much it would improve the
desired inference, divide that benefit by its physical cost, and adapt after
the result is observed.

This page records the mathematical formulation, literature basis, software
architecture, implementation sequence, and validation criteria for adding that
capability to PythonSpinDynamics. The first implementation is deliberately a
small inversion-recovery reference problem. It is useful for algorithm and API
validation; it is not yet an instrument controller or a broad experimental
validation claim.

## Decision-theoretic formulation

Let

- \(d\in\mathcal D\) be controllable design variables such as inversion delay,
  echo spacing, gradient strength, carrier frequency, phase, or repetitions;
- \(\theta\) be parameters of interest such as \(T_1\), \(T_2\), diffusion or
  exchange coefficients, quadrupolar parameters, or concentrations;
- \(\eta\) collect nuisance quantities such as amplitude, baseline, phase,
  field error, noise level, probe response, or drift;
- \(m\) be an optional model index;
- \(y\) be the measured signal or derived observable; and
- \(z=q(\theta,m)\) be the quantity of interest (QoI) that the experiment is
  intended to determine.

For the acquisition history \(H_t=\{(d_k,y_k)\}_{k=1}^t\), the statistical
forward model and posterior are

\[
p(y\mid\theta,\eta,m,d),
\qquad
p(\theta,\eta,m\mid H_t)\propto
p(\theta,\eta,m)\prod_{k=1}^{t}
p(y_k\mid\theta,\eta,m,d_k).
\]

The canonical goal-oriented utility is the conditional mutual information
between the QoI and a future observation:

\[
U_{\mathrm{EIG}}(d\mid H_t)
=I(z;y\mid d,H_t)
=\mathbb E_{y\mid d,H_t}
\left[
D_{\mathrm{KL}}\!\left(
p(z\mid H_t,y,d)\,\|\,p(z\mid H_t)
\right)
\right].
\]

Targeting \(z\), rather than automatically targeting every simulator
parameter, prevents the design from valuing information that is useful only
for irrelevant nuisance directions. For outputs associated with a concrete
decision or accuracy requirement, expected Bayes-risk reduction is even more
general:

\[
U_L(d\mid H_t)=R(H_t)-
\mathbb E_{y\mid d,H_t}R(H_t\cup\{(d,y)\}),
\]

where \(R\) may represent posterior variance, weighted estimation error,
classification loss, or the probability of making an incorrect decision.

The practical one-step acquisition rule is

\[
d_{t+1}=\arg\max_{d\in\mathcal D_{\mathrm{feasible}}}
\frac{U(d\mid H_t)}
{c_{\mathrm{acq}}(d,H_t)+c_{\mathrm{overhead}}(d,H_t)}.
\]

The denominator is *physical experiment time*, not the time required to run a
Python simulation. The fully formal objective is a policy that minimizes
expected cumulative physical time subject to a stopping rule, for example a
credible-interval width or expected loss below a specified threshold. The
information-rate rule is the tractable myopic approximation used first.

## Evidence from prior work

Bayesian experimental design (BED) defines expected information gain (EIG) as
the expected entropy reduction, equivalently mutual information, between an
unknown and a future observation. Sequential BED alternates posterior
inference, design optimization, acquisition, and updating. Its central
computational difficulty is the nested expectation or density-ratio evaluation
inside EIG [1].

Magnetic-resonance work gives several relevant precedents:

1. Waudby, Burridge, and Christodoulou used Fisher-information and G-optimal
   criteria to adapt NMR evolution times and phases on the fly. Measurements
   concentrated at a small set of informative support points and improved
   precision at fixed acquisition count [2]. This is a strong fast baseline,
   but its local linearization at a current parameter estimate does not fully
   represent broad or multimodal uncertainty.
2. Sequential Bayesian design has been demonstrated for Ramsey spin
   measurements with nuisance parameters, non-Gaussian photon-count
   likelihoods, and explicit measurement cost. A Gaussian variance proxy could
   perform worse than random selection when the posterior was nonlinear or
   multimodal, which argues against making a Fisher/Laplace approximation the
   sole backend [3].
3. Adaptive sensing for two-dimensional \(T_1\)-\(T_2\) NMR selected wait
   times using mutual information and obtained better estimation from fewer
   constrained measurements than a fixed schedule [4].
4. Goal-oriented BOED shows why information about a prediction or derived QoI
   can require different designs from information about every model parameter
   [5].
5. EIG rankings can be sensitive to the assumed prior, motivating prior
   sensitivity, model discrepancy, and robustness tests [6].
6. Conditional normalizing flows and learned policies can scale BOED to
   high-dimensional MRI acquisition, but require a substantial offline
   training system. They are a later scaling path, not the initial dependency
   or abstraction for low-dimensional relaxation and diffusion problems [7].

## Fit with the existing library

The current experiment facade already provides immutable experiment
descriptions, compatibility checking, workflow resolution, execution records,
and provenance. Bayesian design should consume those facilities without
changing their meanings.

Two new concepts are required:

- `spin_dynamics.experiment.estimate` predicts *simulation compute time*.
  Adaptive acquisition needs a distinct physical-time model including sequence
  duration, repetitions, recovery, switching, and setup overhead.
- `spin_dynamics.noise.NoiseSpec` generates noise realizations. Bayesian
  updating additionally requires an observation model that can sample and
  evaluate `log_prob(observation, prediction)`.

The long-term package boundary is therefore a sibling layer:

```text
spin_dynamics/design/
    types.py          # Protocols and shared results
    priors.py         # Priors, transforms, joint parameter spaces
    likelihoods.py    # Real/complex Gaussian, correlated, Poisson, Student-t
    posterior.py      # Grid, particle, and Laplace backends
    models.py         # Vectorized predictor plus observation likelihood
    spaces.py         # Candidate and continuous design spaces
    utilities.py      # EIG, Bayes-risk reduction, model discrimination
    costs.py          # Physical acquisition-time models
    constraints.py    # Hardware, duty-cycle, and safety constraints
    adapters.py       # Phase 2 experiment binding and observable extraction
    session.py        # Replayable ask/tell adaptive loop
    diagnostics.py    # ESS, estimator error, calibration, and stopping
    reference_ir.py   # Phase 0 exact-grid inversion-recovery reference
```

An adaptive session should recommend an acquisition and ingest an observation
through an `ask()`/`tell()` interface. It should not directly operate hardware.
In the current package `Experiment.run()` runs a simulator, so conflating that
method with instrument control would make simulations and measurements
ambiguous.

## Phase 0: exact-grid inversion recovery

The reference model is

\[
y=b+A\left(1-2e^{-\tau/T_1}\right)+\epsilon,
\qquad
\epsilon\sim\mathcal N\!\left(0,\sigma^2/r\right),
\]

where \(\tau\) is the selected inversion delay and \(r\) is the number of
independent repetitions averaged into one observation. The discrete joint
prior and posterior cover \(T_1\), amplitude \(A\), baseline \(b\), and the
single-repetition noise \(\sigma\). This deliberately retains nuisance
uncertainty rather than fitting only \(T_1\).

For every candidate delay, the implementation evaluates
\(I(T_1;Y\mid d,H_t)\), marginalizing amplitude, baseline, and noise. Gaussian
expectations are evaluated deterministically by Gauss-Hermite quadrature, and
candidate designs are ranked by nats per physical second. The posterior is an
exact update of the chosen discrete grid; “exact” here refers to the grid
posterior, not to a continuum approximation or to quadrature with zero error.

The physical-time model includes a fixed action overhead and per-repetition
delay, readout, dead time, and recovery. Results are replayable through a JSON
checkpoint containing the grid, posterior weights, design set, measurements,
and cost assumptions.

A minimal adaptive loop is:

```python
import numpy as np

from spin_dynamics.design import (
    IRAcquisitionCost, IRAdaptiveSession, IRDesign,
    IRGridPosterior, IRGridPrior,
)

prior = IRGridPrior(
    t1_seconds=np.geomspace(0.05, 0.8, 25),
    amplitude=np.array([0.9, 1.0, 1.1]),
    baseline=np.array([-0.03, 0.0, 0.03]),
    sigma=np.array([0.04, 0.06]),
)
designs = tuple(IRDesign(delay) for delay in np.geomspace(0.015, 1.6, 12))
session = IRAdaptiveSession(
    IRGridPosterior(prior),
    designs,
    IRAcquisitionCost(readout_seconds=0.01, recovery_seconds=0.08),
)

recommendation = session.ask()
# Acquire recommendation.best.design externally, then supply the averaged value.
session.tell(recommendation.best.design, value=0.137)
print(session.posterior.t1_credible_interval())
```

Run `examples/bayesian_ir_phase0.py` for a synthetic comparison with a fixed
schedule. Its output is a simulation benchmark under the stated model, not a
guarantee for a physical sample.

Phase 0 acceptance criteria are:

- agreement with direct discrete Bayes updates;
- zero target EIG when only one \(T_1\) value is possible;
- nonnegative EIG and deterministic recommendations;
- exact checkpoint/replay behavior;
- calibrated credible intervals in prior-predictive simulations; and
- lower median time to a specified \(T_1\) precision than representative fixed
  schedules under equal likelihood and cost assumptions.

## Subsequent implementation plan

### Phase 1: generic Bayesian design core — implemented

Extract protocols for priors, likelihoods, posterior backends, QoIs, utilities,
costs, constraints, and replayable sessions. Implement NumPy grid and weighted
particle backends, candidate enumeration, EIG, expected posterior risk, score
uncertainty, stopping rules, and deterministic random-number handling. Keep
NumPy as the only core dependency.

The implementation supplies normal, uniform, log-uniform, discrete, and
independent priors; real and circular-complex Gaussian likelihoods; exact
Cartesian grids and weighted particles; systematic resampling; full-state EIG;
QoI variance reduction; common-random-number candidate comparisons; physical
costs; composable constraints; credible-width and posterior-standard-deviation
stopping; JSON checkpoints; and the generic
`examples/bayesian_design_linear.py` workflow. Generic target-only EIG remains
deferred because nuisance marginalization requires a separately validated
conditional-density or nested estimator.

### Phase 2: experiment-facade adapters

Add adapters for `CPMGIRTrain`, `PGSE`, NQR frequency/pulse selection, and ESR
delay/frequency selection. Each adapter must state how latent parameters bind
to a simulation-only `Experiment`, how controllable fields form an acquisition
action, which result is the observable, what the physical action costs, and
which constraints apply. Every proposed action should pass the existing static
experiment planner before it can be scored or returned.

**Implemented 2026-07-16.** `design.adapters` defines typed acquisition
actions, scalar latent-parameter binding, result extraction, simulator-output
caching, physical duration models, and `ExperimentPlanConstraint`.
`make_adapter_session()` installs that constraint automatically. Reference
adapters cover complex echo integrals for `CPMGIRTrain`, deterministic PGSE
echoes, full NQR FIDs under frequency/pulse selection, and ESR Hahn waveforms
under delay/carrier selection. The implementation deliberately keeps
observation noise in the explicit likelihood rather than the simulation-only
experiment template.

### Phase 3: performance and robustness

Add batched simulation, caching, common random numbers, adaptive Monte Carlo
allocation, optional JAX scoring, and a Fisher/Laplace screening backend.
Candidate sets can then be screened cheaply and the leaders rescored with a
particle estimator. Add nuisance marginalization, model mixtures, discrepancy
terms, and prior/model sensitivity reports.

**Implemented 2026-07-16.** Deterministic PGSE adapters now evaluate posterior
particles as one vectorized batch while retaining static planner validation and
facade parity. `PolynomialSurrogatePredictor` supports ordinary or
physics-informed joint features and reports held-out RMSE, normalized RMSE,
maximum error, and correlation. `LaplaceInformationGain` supplies NumPy or
optional-JAX Fisher/Laplace screening; `TwoStageDesignSession` refines only the
screened leaders with the requested particle utility; and
`AdaptiveMonteCarloUtility` stops nested sampling when its standard error is
resolved. `ExpectedTargetInformationGain` marginalizes nuisance particles for
discrete targets. Explicit discrepancy likelihoods, posterior model-mixture
dispatch, and named prior/model sensitivity reports cover the initial
robustness layer. Runtime plots record exact-parity PGSE batching and held-out
validated CPMG-IR surrogate screening; broader workflow batching and continuous
target-information estimators remain open.

### End-to-end Phase 2 adapter benchmark — implemented

The four Phase 2 adapters now have a common paired prior-predictive benchmark.
For each synthetic truth and noise stream it compares an adaptive policy, a
declared fixed coverage schedule, and a prior-ranked schedule that is frozen
before observations arrive. All policies share the exact adapter predictor,
likelihood, posterior support, target-oriented variance utility, physical cost,
credible-interval stopping rule, and maximum acquisition count.

`CandidatePredictionTable` evaluates the finite particle/action support once
and is then an exact lookup, not an approximate surrogate. Deterministic PGSE
uses its Phase 3 batch implementation while the other adapters amortize exact
facade simulations across trials. The result records physical acquisition
seconds, adaptive planning wall time, and one-time table construction
separately. The 96-trial reference plot and JSON output cover CPMG-IR T1,
PGSE diffusion, NQR site frequency, and ESR Hahn T2. They show improved target
success or physical efficiency for adaptive acquisition in all four stated
synthetic problems, while retaining cases where a fixed policy wins an
individual metric.

This is equal-model evidence about the implemented decision algorithm. It is
not evidence of time savings on an instrument, calibration under model
discrepancy, or superiority for arbitrary priors and action spaces. Those
questions remain physical-validation work beyond the Phase 4 orchestration
reference.

### Phase 4: batch and live operation

Support small batches when adaptation cannot occur after every scan, explicit
planner latency, atomic checkpoints, external instrument adapter examples,
rejected acquisitions, and audit logs containing posterior summaries,
candidate scores, costs, constraints, and selection reasons.

**Implemented 2026-07-16.** `LiveDesignSession` wraps the generic adaptive
session with a synchronous `DesignInstrument` protocol. It ranks once per
small batch, selects distinct feasible actions, measures planning wall time
against an optional latency budget, and separately records instrument-reported
physical time and non-overlapped operational time. Static planner rejections
remain in candidate audits; instrument rejections do not update the posterior,
and non-retryable actions are excluded from later batches.

Every pending batch is written with `save_design_state_atomic()` before the
instrument is called. The attempt itself is also checkpointed before external
I/O, so a lost acknowledgement restores as an ambiguous pending acquisition.
Automatic replay is refused until the caller supplies recovered outcomes or
explicitly authorizes a retry. Candidate utilities and uncertainty, costs,
constraint messages, selection reasons, request IDs, outcomes, timing, and
before/after QoI summaries are retained in an atomically replaced JSON audit.
The runnable CPMG-IR example includes a synthetic external adapter and interlock
rejection while keeping the hardware boundary replaceable.

This implementation is a synchronous orchestration reference, not a vendor
driver, safety controller, distributed scheduler, or non-myopic batch design
algorithm. Real deployments must add instrument-specific authentication,
timeouts, calibration, archive reconciliation, and idempotency guarantees.

### Phase 5: advanced policies

Consider continuous stochastic-gradient design, multi-step lookahead,
amortized posterior estimators, learned non-myopic policies, and high-dimensional
spatial quantities only after the reference workflows have strong calibration
and equal-budget evidence.

## Validation strategy

Evaluation must compare policies at equal *physical acquisition time*, not
merely verify that an estimated EIG increased. The primary metrics are:

- distribution of time to target precision or decision loss;
- error and credible-interval coverage at equal time;
- posterior calibration and simulation-based calibration ranks;
- sensitivity to priors, nuisance parameters, and noise assumptions;
- failure under model discrepancy;
- utility regret relative to exhaustive small reference problems; and
- planner computation time relative to acquisition time.

Fixed uniform, fixed logarithmic, random, Fisher-optimal, and Bayesian-adaptive
policies should be retained as named baselines. A regression test verifies the
implemented calculation; it does not constitute experimental validation.

## References

1. T. Rainforth, A. Foster, D. R. Ivanova, and F. Bickford Smith, “Modern
   Bayesian experimental design,” *Statistical Science* 39, 100–114 (2024),
   [doi:10.1214/23-STS915](https://doi.org/10.1214/23-STS915).
2. C. A. Waudby, C. Burridge, and J. Christodoulou, “Optimal design of
   adaptively sampled NMR experiments for measurement of methyl group dynamics
   with application to a ribosome-nascent chain complex,” *Journal of Magnetic
   Resonance* 326, 106937 (2021),
   [doi:10.1016/j.jmr.2021.106937](https://doi.org/10.1016/j.jmr.2021.106937).
3. R. D. McMichael, S. Dushenko, and S. M. Blakley, “Sequential Bayesian
   experiment design for adaptive Ramsey sequence measurements,” *Journal of
   Applied Physics* 130, 144401
   (2021), [doi:10.1063/5.0055630](https://doi.org/10.1063/5.0055630).
4. S. Pitawala, P. D. Teal, and M. Frean, “Adaptive sensing for NMR
   measurements,” *Journal of Magnetic Resonance* 386, 108055 (2026),
   [doi:10.1016/j.jmr.2026.108055](https://doi.org/10.1016/j.jmr.2026.108055).
5. S. Zhong, W. Shen, T. Catanach, and X. Huan, “Goal-oriented Bayesian optimal
   experimental design for nonlinear models using Markov chain Monte Carlo,”
   (2024), [arXiv:2403.18072](https://arxiv.org/abs/2403.18072).
6. J. Go and T. Isaac, “Robust expected information gain for optimal Bayesian
   experimental design using ambiguity sets,” *Proceedings of Machine Learning
   Research* 180, 728–737 (2022),
   [paper and code links](https://proceedings.mlr.press/v180/go22a.html).
7. R. Orozco, F. J. Herrmann, and P. Chen, “Probabilistic Bayesian optimal
   experimental design using conditional normalizing flows” (2024),
   [arXiv:2402.18337](https://arxiv.org/abs/2402.18337).
