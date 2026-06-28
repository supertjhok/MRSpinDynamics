# Build and Review Notes

This document describes how the NQR database is generated, reviewed, and served
locally. It is written for maintainers who need to rebuild the database or audit
how reviewed Landolt-Bornstein PDF data becomes canonical compound/site/line
data.

## Repository Layout

- `schema/` - JSON schema files and future migration definitions.
- `data/normalized/` - generated JSON Lines tables.
- `data/exports/nqr.sqlite` - generated SQLite database used by the GUIs.
- `data/review/landolt_review_decisions.jsonl` - append-only review decisions.
- `scripts/build_database.py` - top-level rebuild entry point.
- `scripts/build_cwru_database.py` - current importer and exporter.
- `app/review_server.py` - local Landolt review GUI server.
- `app/explorer_server.py` - local database explorer GUI server.

The generated normalized files and SQLite export are committed so the database
can be used without rerunning OCR/PDF extraction on every checkout.

## Rebuild Command

Run from the repository root:

```powershell
python NQRDatabase/scripts/build_database.py
```

The build reads local source material from `References/NQR Data`, replays the
latest review decisions, and regenerates:

- `data/exports/nqr.sqlite`
- `data/normalized/*.jsonl`
- `data/normalized/line_records.jsonl`

The current build reports:

- `sources=41`
- `compounds=184`
- `samples=250`
- `sites=548`
- `lines=923`
- `literature_references=117`
- `reference_links=1268`
- `landolt_compound_entries=166`
- `landolt_measurement_sets=207`
- `landolt_frequency_records=654`
- `landolt_qcc_eta_records=228`
- `landolt_review_queue=166`

## Source Imports

The importer currently combines four source families.

1. Archived web pages from an earlier NQR database associated with Case Western
   Reserve University and the University of Florida. These pages supply
   compound, isotope/site, line-frequency, relaxation, linewidth, temperature,
   and source fields where the original pages recorded them.
2. U.S. Navy / Naval Research Laboratory NQR data tables. The summary tables
   supply line and site records, while associated source notes are converted
   into `literature_references` and `reference_links`.
3. King's College experimental notes. These are manually structured into the
   same compound/sample/site/line model and preserve acquisition notes where
   available.
4. Landolt-Bornstein NQR excerpts. These are OCR/layout-derived and therefore
   pass through staging tables and the review GUI before promotion.

Every source collection is represented in `sources`. Paper-level citations are
stored in `literature_references`; links to compounds, sites, and lines are
stored in `reference_links`.

## Canonical Tables

The canonical model is:

- `compounds` - one record per canonical compound name/formula/category.
- `samples` - one record per reported material condition, often including
  method and temperature.
- `sites` - isotope/site records, including quadrupole coupling constants and
  eta values when available.
- `lines` - observed transition frequencies and line-level metadata.
- `literature_references` - paper or source citations.
- `reference_links` - provenance links from references to compounds, sites, and
  lines.
- `sources` - local source files and source collections.

Canonical numeric frequencies and quadrupole coupling constants are stored in
kHz. Source strings are retained in `*_original` fields and JSON
`original_record` payloads.

The denormalized `data/normalized/line_records.jsonl` file is intended for AI
tools, search indexes, and lightweight applications that want one record per
line observation without joining the full schema.

## Landolt Staging Tables

Landolt PDF-derived material is also preserved in staging tables:

- `nqr_transition_equations`
- `landolt_column_definitions`
- `landolt_page_extracts`
- `landolt_compound_entries`
- `landolt_reference_codes`
- `landolt_measurement_sets`
- `landolt_frequency_records`
- `landolt_qcc_eta_records`
- `landolt_review_queue`

These tables retain raw row text, footnote text, parsed names/formulas/CAS
numbers, measurement method labels, source temperatures, reference codes,
frequencies, quadrupole-coupling constants, eta values, rendered crop paths, and
review status.

## Review GUI

Start the Landolt review interface from the repository root:

```powershell
python NQRDatabase/app/review_server.py
```

Open `http://127.0.0.1:8765`.

The GUI reads `data/exports/nqr.sqlite`, displays `landolt_review_queue`, and
saves accepted/rejected decisions to:

```text
data/review/landolt_review_decisions.jsonl
```

The review workflow is:

1. Compare the parsed row with the PDF crop.
2. Correct identity fields, measurement sets, frequencies, and
   quadrupole-coupling/eta lists as needed.
3. Save edits.
4. Accept the record once the reviewed values match the source evidence.

The decision file is append-only. During rebuild, the latest decision for each
`review_id` wins.

## Landolt Promotion Rules

Accepted Landolt review decisions are promoted into canonical records during
the build.

Some compounds span more than one PDF page. Before promotion, accepted review
records are grouped by source, table number, and substance number. Records in
the same group are merged so that continuation-page frequencies,
coupling/eta pairs, references, names, formulas, and raw source text become one
canonical promoted entry. Less-complete duplicate continuation records are not
promoted as separate compounds.

Within a Landolt measurement condition, frequency values and coupling/eta
values are treated as independent lists. The build does not infer a one-to-one
assignment between the two lists. Each accepted measurement set becomes one
canonical sample:

- line frequencies are stored under a synthetic unassigned frequency-list site;
- each coupling/eta pair becomes its own site;
- coupling/eta sites use `assignment_confidence =
  source_reported_unassigned_to_lines`.

When the same ordered frequency list is reported at multiple temperatures for a
compound and method, the importer labels corresponding ordered positions as a
temperature series. It fits a simple linear slope and stores it in:

- `lines.transition_label`
- `lines.dnu_dt_khz_per_c`
- `lines.dnu_dt_original`

This coefficient is a convenience annotation for obvious repeated-temperature
series, not a claim that every line has been manually assigned to a physical
transition.

## Landolt Method Labels

Landolt method labels are carried through from the source tables:

- `C` - continuous wave method.
- `D` - double resonance method.
- `P` - pulse method.
- `M` - NMR method.
- `E` - other methods.
- `X` - method not described in the original paper or not recorded in the
  database at the early stage.

Temperature tokens such as `RT`, `RTemp`, and `R.Temp` mean room temperature.
They are preserved as source text and do not imply an exact numeric temperature
unless the table or reviewer provides one.

## Explorer GUI

Start the human-facing database explorer from the repository root:

```powershell
python NQRDatabase/app/explorer_server.py
```

Open `http://127.0.0.1:8766`.

The explorer serves the generated SQLite export. It provides:

- an overview page explaining database terminology and assumptions;
- search by compound name, formula, CAS number, and notes;
- filters by category, isotope, source family, and frequency range;
- compound detail views with samples, sites, lines, references, and sources;
- spectrum plots with site-aware coloring when site assignments are available;
- displayed quadrupole-coupling and eta symbols in the UI;
- browser-side PubChem structure images when a compound can be matched.

If the database is rebuilt while the explorer is running, restart the explorer
so it opens the refreshed SQLite export.

## Commit Checklist

After review or importer changes:

1. Rebuild with `python NQRDatabase/scripts/build_database.py`.
2. Spot-check affected compounds in `data/exports/nqr.sqlite` or the explorer.
3. Confirm the review queue counts still match the intended decision state.
4. Stage the changed normalized files, SQLite export, review decision log, and
   importer/docs changes.
5. Leave scratch files, local logs, caches, and unrelated workspace files
   unstaged.
