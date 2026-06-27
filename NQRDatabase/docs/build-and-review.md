# NQR Database

Working area for a combined nuclear quadrupole resonance database built from the
local reference material in `References/NQR Data`.

## Layout

- `schema/` - validation schemas and future database migrations.
- `data/raw/` - source extracts copied or generated from reference material.
- `data/normalized/` - validated, normalized records for import/export.
- `data/exports/` - generated SQLite, JSONL, CSV, or UI-facing exports.
- `scripts/` - import, validation, and export utilities.
- `app/` - local review and human-facing interface utilities.

The intended canonical model is a relational database with lossless JSONL
exports: compounds, samples, sites, lines, pulse responses, and sources.
Compound records include both source-style `formula` and UI-oriented
`conventional_formula` fields. Prefer `conventional_formula` for chemical
lookup, structure-rendering queries, and display.

## Current Build

The current build imports:

- Saved CWRU/UF Google Sites pages from `References/NQR Data/CWRU NQR Database`.
- The compact `NQR Database.pdf` as a fallback source for entries missing from
  the local HTML capture.
- Navy/NRL `NQR_data_tables_summary.pdf` and
  `NQR_data_tables_summary2.pdf` site/line summaries.
- King's College notes for melamine, metformin HCl, and paracetamol.
- Landolt-Bornstein NQR excerpts from `References/NQR Data/nqr_data`,
  including source page text, column definitions, transition-frequency
  equations, staged nitrogen table entries, and Table 9 reference codes.

Run from the repository root:

```powershell
python NQRDatabase/scripts/build_database.py
```

Generated artifacts:

- `data/exports/nqr.sqlite` - SQLite database.
- `data/normalized/*.jsonl` - normalized JSONL tables.
- `data/normalized/line_records.jsonl` - denormalized AI-friendly line records.

Navy/NRL citations and source notes are stored in:

- `literature_references`
- `reference_links`
- `data/normalized/literature_references.jsonl`
- `data/normalized/reference_links.jsonl`

Navy line records in `line_records.jsonl` also include linked references in a
`references` array.

Landolt material is stored in separate staging tables because the PDFs are
OCR/layout-derived and should be checked against the page images. Accepted
review decisions are also promoted into the canonical
`compounds`/`samples`/`sites`/`lines`/`literature_references` records:

- `nqr_transition_equations` / `nqr_transition_equations.jsonl`
- `landolt_column_definitions` / `landolt_column_definitions.jsonl`
- `landolt_page_extracts` / `landolt_page_extracts.jsonl`
- `landolt_compound_entries` / `landolt_compound_entries.jsonl`
- `landolt_reference_codes` / `landolt_reference_codes.jsonl`
- `landolt_measurement_sets` / `landolt_measurement_sets.jsonl`
- `landolt_frequency_records` / `landolt_frequency_records.jsonl`
- `landolt_qcc_eta_records` / `landolt_qcc_eta_records.jsonl`
- `landolt_review_queue` / `landolt_review_queue.jsonl`

The current Landolt staging import captures 104 nitrogen-table entries: 69
from Table 4 and 35 from Table 9. It keeps raw formulas, frequencies, Q.C.C.,
eta, remarks, reference codes, footnote names, CAS numbers where parsed, and
the original row/footnote text for audit.

For Landolt rows, method/temperature/reference groups are stored as
`landolt_measurement_sets`. Each set has its own frequency list and Q.C.C./eta
pair list. In canonical records, each accepted measurement set becomes one
sample. Its frequencies are stored as lines under a synthetic
`unassigned_frequency_list` site, while each Q.C.C./eta pair is stored as a
separate site with `assignment_confidence` set to
`source_reported_unassigned_to_lines`. The database does not infer an
element-wise assignment between frequency values and Q.C.C./eta values within a
set.

Landolt source frequencies and Q.C.C. values are in MHz. Canonical
`frequency_khz` and `qcc_khz` values are converted to kHz, while the source text
is preserved in `frequency_original`, site `original_record`, and the staging
tables.

Landolt method labels are defined as:

- `C` - Continuous wave method.
- `D` - Double resonance method.
- `P` - Pulse method.
- `M` - NMR method.
- `E` - Other methods.
- `X` - Method not described in the original paper or not recorded in the
  database at the early stage.

For Landolt temperatures, source tokens such as `RT`, `RTemp`, and `R.Temp`
mean room temperature. They are preserved as source text and do not imply an
exact numeric temperature unless a reviewer adds one explicitly.

The Landolt review queue assigns a simple priority from OCR/coverage flags and
points to a rendered PNG crop under `data/review/landolt_crops`. These crops
combine the table row with the associated footnote region so reviewers can
compare parsed values against the source image before accepting or correcting a
row. Rebuilds replay the latest decisions from
`data/review/landolt_review_decisions.jsonl`, so the SQLite review status and
canonical promoted records stay in sync.

## Landolt Review GUI

Run the local review interface from the repository root:

```powershell
python NQRDatabase/app/review_server.py
```

Then open `http://127.0.0.1:8765`. The GUI reads the SQLite export, shows the
review queue with crop images, and saves review decisions to:

- `data/review/landolt_review_decisions.jsonl`

The SQLite `landolt_review_queue` status is also updated so filters reflect the
current review session.

For rows with multiple method/temperature/reference groups, use the Measurement
Sets section to review each group separately. Frequency records and Q.C.C./eta
records are still not assigned to each other in this review workflow. The raw
frequency/Q.C.C./eta lists remain unchanged as source evidence.
