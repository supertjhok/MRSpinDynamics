# mr-integration

Cross-project integration layer for **MRSpinDynamics**. It connects the three
otherwise-independent subprojects into a single
**predict → simulate → validate** loop:

```
   QuadrupolarDFT                 NQRDatabase
 (ab initio C_Q, eta)        (measured frequencies)
         |                            |
         v                            |
  PythonSpinDynamics                  |
   (NQR simulation)                   |
         |                            |
         +---------> compare <--------+
```

## What it does

- **Convention bridge** (`conversions`): the validated mapping between the DFT
  quadrupolar coupling constant `C_Q = eQVzz/h` and the simulator's
  `quadrupole_frequency_hz` (`nu_Q`):

  ```
  nu_Q = (3/4) C_Q   (spin-1)
  nu_Q = (1/2) C_Q   (spin-3/2)
  ```

  Builds a `spin_dynamics.nqr.QuadrupolarSite` from a `(C_Q, eta, spin)` triple
  or directly from a `quadrupolar_dft.AbinitEFGRecord`.

- **Self-consistency check** (`cross_validation`): runs DFT-derived parameters
  through `spin_dynamics` and confirms the resulting NQR lines match
  `quadrupolar_dft.nqr_frequencies_hz` — two independent Hamiltonian
  implementations agreeing (to < 1 Hz) is the real proof the bridge is correct.

- **Database lookup** (`database`): reads measured lines for a compound/isotope
  from the `NQRDatabase` SQLite export (read-only; no dependency on the
  `NQRDatabase` Python code).

- **End-to-end report** (`pipeline`): pairs predicted lines against measured
  lines and reports per-line and RMS differences.

- **Database self-consistency validator** (`database_validation`): runs the
  *whole* database through the simulator. Each curated site stores both
  `(qcc, eta)` and its measured lines; those must agree. `validate_database()`
  checks every supported site and sorts by discrepancy, surfacing likely
  OCR/transcription errors. For spin-1 sites it back-solves the parameters
  *implied by the lines*, localizing whether `qcc` or `eta` is wrong.

- **Flag overlay writer** (`flag_export`): writes the validator's verdicts back
  into the database as a `site_consistency_flags` table plus a matching
  `site_consistency_flags.jsonl` export. The NQR explorer reads this overlay and
  shows a per-site badge (flagged / simulator-verified) with the diagnostic
  detail. The overlay is a *derived* product, regenerated after a database
  build; the build itself never depends on the simulator.

- **Landolt review flagging** (`landolt_validation` + `landolt_review_export`):
  the canonical `sites` table holds parameters and lines together, but the
  Landolt import splits them — each measurement set has an independent list of
  frequencies and of `(QCC, eta)` pairs. `validate_landolt_sets` predicts the
  two strong lines (`nu_+`, `nu_-`) for each pair and checks they appear among
  the tabulated frequencies (the weak `nu_0` line is ignored; extra measured
  lines are harmless). `write_landolt_review_flags` routes inconsistencies into
  `landolt_review_queue` — adding a `quad_consistency_mismatch` issue flag,
  raising the entry's priority, and recording the diagnostic in a
  `landolt_consistency_flags` table/JSONL. The review GUI shows the flag and a
  diagnostic banner.

## Install

Requires the two sibling packages on the path:

```bash
pip install -e ./PythonSpinDynamics -e ./QuadrupolarDFT -e ./integration
```

## Try it

```bash
python integration/examples/nano2_dft_vs_measured.py
```

NaNO₂ ¹⁴N is the seed case — it has DFT, simulated, and measured values on all
three sides. Feeding the **literature** `C_Q`/`eta` through the loop reproduces
the measured 1.038 / 3.604 / 4.642 MHz lines to < 1 kHz; the **starter DFT
geometry** lands ~600 kHz off because it underestimates `eta` (a known limit of
that unrelaxed structure, documented in `QuadrupolarDFT`).

For the database-wide consistency scan:

```bash
python integration/examples/database_consistency.py
```

On the current export this checks 61 sites; 56 are self-consistent and 5 are
flagged — e.g. a ³⁵Cl line that implies a ~4 MHz-larger `C_Q` than stored, and
several ¹⁴N sites whose stored `eta` disagrees with their lines (the QCC matches
to < 1 kHz, so the error is localized to `eta`).

To write the flags into the database so the explorer shows them:

```bash
python integration/scripts/write_consistency_flags.py
```

Run this after each database build. Then start the explorer
(`python NQRDatabase/app/explorer_server.py`) — each site with quadrupolar
parameters and lines shows a flagged or simulator-verified badge, and the
compound header summarizes how many of its sites are flagged.

To flag inconsistent Landolt entries for re-review:

```bash
python integration/scripts/write_landolt_review_flags.py
```

This adds a `quad_consistency_mismatch` flag and raises priority on the affected
`landolt_review_queue` rows. Start the review GUI
(`python NQRDatabase/app/review_server.py`) and the flagged entries sort to the
top with a diagnostic banner. On the current export, 26 of 141 checked Landolt
sets are flagged (e.g. a `QCC` OCR error of 313 MHz instead of ~3.13 MHz).

## Tests

```bash
cd integration && python -m unittest discover -s tests
```

Database-backed tests skip automatically if `NQRDatabase/data/exports/nqr.sqlite`
is absent.
