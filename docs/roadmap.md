# MRSpinDynamics — Repository Survey and Roadmap

_Last updated: 2026-06-28_

This is a workspace-level survey and forward plan. Subproject-specific
status lives in each subproject's own docs (e.g.
`PythonSpinDynamics/docs/python_api/known_gaps.md`); this document is about
the **whole repository** and especially the connections *between* the
subprojects.

## 1. Current shape

Four components of very unequal maturity:

| Subproject | Scale | Maturity | Role |
|---|---|---|---|
| `PythonSpinDynamics/` | ~65k LOC, 38 test files, CI matrix (3 Py × 2 OS) | Production-grade | NMR/NQR/ESR simulation — the crown jewel |
| `NQRDatabase/` | ~4.5k LOC, ~900 measured lines, 184 compounds | Solid, app-backed | Curated **measured** NQR frequencies + provenance |
| `QuadrupolarDFT/` | ~1k LOC | Early/thin | Ab initio EFG → C_Q, η, **predicted** NQR frequencies |
| `MATLABSpinDynamics/` | large | Frozen reference | Validation oracle for the Python port |

`PythonSpinDynamics` is near feature-complete against its MATLAB origin; its
`known_gaps.md` reads as a stabilization list, not a missing-feature list. So
the highest-value work is no longer "port more MATLAB" — it is **structural and
cross-cutting**.

## 2. The headline gap: the subprojects don't talk to each other

The repository contains the three pieces of a complete **predict → simulate →
validate** loop, but (as of this survey) **zero cross-project imports** connect
them:

- `QuadrupolarDFT` computes a *predicted* NQR frequency from first principles
  (EFG → C_Q, η → ν).
- `PythonSpinDynamics` can *simulate* the full pulsed NQR response given
  (C_Q, η, spin).
- `NQRDatabase` holds the *measured* ν for 184 compounds, with citations.

Closing this loop is the single highest-science-value, modest-code improvement
available. A ready-made first case already exists on all three sides: **NaNO₂
¹⁴N**.

- DFT (ICSD 82857 run): C_Q ≈ −5.034 MHz, η ≈ 0.112 → lines 0.282, 3.635,
  3.916 MHz.
- Database (literature): QCC = 5.497 MHz, η = 0.378 → lines 1.038, 3.604,
  4.642 MHz.
- Simulator: given (C_Q, η, spin), diagonalizes to either line set.

### Convention bridge (validated)

The simulator parameterizes a site by `quadrupole_frequency_hz` (ν_Q, the η = 0
transition); the DFT side reports C_Q = eQVzz/h. These are linked by

```
ν_Q = C_Q · d / (4 I (2I − 1)),   d = 3 (spin-1), 6 (spin-3/2)
     = (3/4) C_Q   for spin-1
     = (1/2) C_Q   for spin-3/2
```

Verified end-to-end: feeding the NaNO₂ ¹⁴N C_Q/η through the conversion into
`spin_dynamics.nqr.diagonalize_site` reproduces `quadrupolar_dft.
nqr_frequencies_hz` to < 1 Hz (two independent Hamiltonian implementations
agreeing is the actual proof). This is implemented in the new `integration/`
package — see section 5.

## 3. Other technical gaps (workspace level)

**Performance / scale** (explicitly deferred in `known_gaps.md`)
- No compiled or GPU backend. ~65k LOC of dense NumPy isochromat propagation.
  A JAX/Numba engine would buy both speed *and* autodiff (the optimization
  module currently uses pattern search + SciPy).
- NQR module is dense-matrices-only — blocks spin ≥ 5/2 and multi-band solvers.

**Packaging / distribution**
- `version = "0.0.0"`, "Development Status :: 3 - Alpha", not on PyPI, no
  published API-docs site (the `generate_api_reference.py` + MkDocs scaffolding
  is half-built). This is the limiting factor on adoption/citation; a JOSS
  paper is realistic given the validation depth.
- `QuadrupolarDFT/runs/**` commits raw ABINIT binary outputs (`.nc`, `_DDB`,
  `.abo0001`…) into git — should be gitignored before history bloats.

**Test / CI**
- Only `PythonSpinDynamics` has CI. `QuadrupolarDFT` and `NQRDatabase` have
  none.
- No coverage measurement; benchmarks exist but aren't gated, so perf
  regressions are invisible. ESR (newest module) has the thinnest test surface.

**Physics depth** (next-frontier, from the science-impact roadmap)
- q-space / averaged-propagator pore-size (diffusion-diffraction) — unstarted;
  the PGSTE/walker machinery already exists, so this is mostly new analysis.
- NQR/NMR relaxation is phenomenological (T1/T2/T2e), not microscopic
  Redfield/dipolar.

## 4. Ranked opportunities (impact ÷ effort)

1. **Close the DFT → sim → DB loop.** Highest science value, modest code.
   Started here as the `integration/` package; NaNO₂ is the seed case.
2. **q-space diffusion-diffraction** (roadmap #4) — reuses existing PGSTE/walker
   infrastructure.
3. **JAX/Numba isochromat backend** — unlocks speed *and* autodiff pulse
   optimization. Highest engineering payoff.
4. **Publish.** Version bump → PyPI → MkDocs site → JOSS.
5. **Database enrichment from DFT** — a "predicted vs measured" column in the
   NQR explorer UI. Visually striking, directly useful.
6. **Repo hygiene** — gitignore ABINIT binaries; add CI for the two uncovered
   subprojects.

## 5. The `integration/` layer (in progress)

A new top-level package, `mr_integration`, that depends on both
`spin_dynamics` and `quadrupolar_dft` and reads the NQR SQLite export. It is the
concrete realization of opportunity #1.

Scope of the first increment:

- `conversions` — validated C_Q ↔ ν_Q mapping and a `quadrupolar_site_from_cq`
  / `quadrupolar_site_from_efg_record` builder that returns a
  `spin_dynamics.nqr.QuadrupolarSite`.
- `cross_validation` — run DFT-derived parameters through the simulator and
  assert self-consistency with `quadrupolar_dft.nqr_frequencies_hz`.
- `database` — query measured lines for a compound/isotope from `nqr.sqlite`.
- `pipeline` — end-to-end **predicted vs measured** comparison report.
- `examples/nano2_dft_vs_measured.py` — the seed demonstration.

Second increment (done): **database self-consistency validator**
(`database_validation`). Each curated site stores both `(qcc, eta)` and its
measured lines; the simulator must reproduce one from the other.
`validate_database()` scans every supported site (currently 61: ¹⁴N spin-1 plus
a few spin-3/2) and sorts by discrepancy. On the current export, 56/61 are
self-consistent and 5 are flagged as likely transcription/OCR errors — e.g. a
³⁵Cl Cladribine line implying a ~4 MHz-larger `C_Q` than stored, and several
¹⁴N sites whose stored `eta` disagrees with their lines (the spin-1
implied-parameter back-solve localizes the error to `eta`, since QCC matches to
< 1 kHz). Example: `integration/examples/database_consistency.py`.

Third increment (done): **explorer flag overlay**. `flag_export` writes a
`site_consistency_flags` table (and JSONL) into the database; the NQR explorer
(`explorer_server.py` + `explorer_static`) reads it and shows a per-site badge
(flagged / simulator-verified) with the diagnostic detail, plus a compound-level
"N sites flagged" summary chip. Regenerate after a build with
`integration/scripts/write_consistency_flags.py`. The overlay is derived, so the
database build never depends on the simulator, and the explorer degrades
gracefully if the table is absent.

Fourth increment (done): **Landolt review-queue flagging**. The Landolt import
splits frequencies and `(QCC, eta)` into independent lists per measurement set,
so the site-level validator can't see them together. `landolt_validation`
predicts the two strong lines (`nu_+`, `nu_-`) per `(QCC, eta)` pair and checks
they appear among the tabulated frequencies; `landolt_review_export` routes
mismatches into `landolt_review_queue` (a `quad_consistency_mismatch` issue
flag + raised priority + a `landolt_consistency_flags` detail table/JSONL), and
the review GUI shows a diagnostic banner. On the current export, 26 of 141
checked Landolt sets are flagged — e.g. a `QCC` OCR error (313 MHz for a line at
~2.7 MHz). Regenerate with
`integration/scripts/write_landolt_review_flags.py`.

Later increments: feed DFT η/C_Q distributions into the simulator's
EFG-broadening models; widen DFT coverage so the predict-vs-measured loop runs
over more than NaNO₂; extend Landolt checking beyond ¹⁴N once spin ≥ 5/2 lands.
