# NQR Spectra Database

This folder contains a curated nuclear quadrupole resonance (NQR) spectra
database. NQR is a radio-frequency spectroscopy method for nuclei with electric
quadrupole moments, including nitrogen-14, chlorine-35, potassium-39, bromine,
and iodine nuclei. The database is meant to preserve both the values useful to
scientific users and the source evidence needed to audit those values.

The project has two goals:

- provide AI-friendly exports that are easy to parse, search, and cite;
- provide a human-facing browser interface for exploring compounds, spectra,
  sites, references, and source provenance.

There is no single standard online NQR spectra database. This project combines
the local source material under `References/NQR Data` into reproducible SQLite
and JSON Lines exports, plus local web interfaces for review and exploration.

## Current Build

The current generated database contains:

- 184 compounds
- 250 samples or reported measurement conditions
- 548 isotope/site records
- 923 NQR line-frequency records
- 117 literature-reference records
- 1268 links from compounds, sites, and lines to references

The main reusable outputs are:

- `data/exports/nqr.sqlite` - SQLite database with normalized tables.
- `data/normalized/*.jsonl` - one JSON Lines file per normalized table.
- `data/normalized/line_records.jsonl` - denormalized line records for AI tools,
  search indexes, and lightweight applications.

## Local Interfaces

The database includes two small local web applications.

### Explorer GUI

The explorer is the main human-facing interface. Start it from the repository
root:

```powershell
python NQRDatabase/app/explorer_server.py
```

Then open `http://127.0.0.1:8766`.

The explorer supports compound search, category/isotope/source/frequency
filters, compound detail pages, line plots, measurement tables, source and
reference links, and an overview page explaining the database terms. Where a
compound name, formula, or CAS number can be matched, the browser attempts to
display a 2D structure image from PubChem; otherwise it falls back to the stored
formula.

### Landolt Review GUI

The review interface is for checking OCR/layout-derived rows from the
Landolt-Bornstein PDFs. Start it from the repository root:

```powershell
python NQRDatabase/app/review_server.py
```

Then open `http://127.0.0.1:8765`.

The review GUI displays rendered PDF crops, parsed identity fields, measurement
sets, frequency lists, and quadrupole-coupling/asymmetry lists. It writes the
latest review decisions to `data/review/landolt_review_decisions.jsonl`; the
build replays those decisions when generating the canonical database.

## Data Sources

Every imported source is represented in the `sources` table and
`data/normalized/sources.jsonl`. The major source collections are:

- Archived pages from an earlier online NQR database associated with Case
  Western Reserve University and the University of Florida. The local archive
  contains saved Google Sites pages captured on 2020-10-11 in
  `References/NQR Data/CWRU NQR Database/`, plus `References/NQR Data/NQR
  Database.pdf`. These pages currently supply 120 line records.
- U.S. Navy / Naval Research Laboratory NQR data tables. Local copies are in
  `References/NQR Data/NQRdatabase/NQR_Data_Tables.chm` and
  `References/NQR Data/NQRdatabase/nqr_tables/`. Imported PDF tabulations
  include `NQR_data_tables_summary.pdf`, `NQR_data_tables_summary2.pdf`, and
  `NQR_data_tables_all.pdf`; they currently supply 77 line records and
  compound-level citation notes.
- King's College experimental NQR notes in
  `References/NQR Data/NQRdatabase/kings_college_database/`. The imported notes
  cover melamine, metformin HCl, paracetamol, and related coil/population
  transfer measurements; they currently supply 25 line records.
- H. Chihara and N. Nakamura, *Nuclear Quadrupole Resonance Spectroscopy Data*,
  Landolt-Bornstein, Condensed Matter series, edited by K.-H. Hellwege and
  A. M. Hellwege. Local excerpts are in `References/NQR Data/nqr_data/`,
  including nitrogen tables, transition-frequency formula pages, and reference
  code pages. Reviewed Landolt nitrogen entries currently supply 701 line
  records.

Specific paper citations from the Navy/NRL and Landolt sources are stored in
`literature_references` and connected to compounds, sites, and lines through
`reference_links`.

## Data Model

The canonical database is organized around:

- `compounds` - names, formulas, display formulas, categories, and notes.
- `samples` - measured materials or conditions, including temperature when
  known.
- `sites` - isotope/site information, quadrupole coupling constants, and eta
  values.
- `lines` - resonance frequencies, source temperatures, linewidths, relaxation
  values, forms, and temperature coefficients when available.
- `literature_references` and `reference_links` - paper-level provenance.
- `sources` - local files or source collections used by the importer.

Canonical frequency and quadrupole-coupling fields are stored in kHz. Source
strings are retained in `*_original` fields and in JSON `original_record`
payloads so the import can be audited later.

## Landolt Semantics

The Landolt-Bornstein PDFs are OCR/layout-derived, so accepted rows are promoted
only through the review decision log. Some compounds span multiple PDF pages;
the build merges accepted continuation-page records by source, table, and
substance number before promotion. Less-complete duplicate continuation records
are not promoted separately.

Landolt rows often report independent lists of line frequencies and
quadrupole-coupling/asymmetry pairs for the same measurement condition. The
database deliberately does not infer a one-to-one assignment between those two
lists. In canonical records, each accepted Landolt measurement set becomes one
sample. Its frequencies are stored under an unassigned frequency-list site, and
each coupling/eta pair is stored as a separate site with
`assignment_confidence` set to `source_reported_unassigned_to_lines`.

When a Landolt entry gives the same ordered line list at multiple temperatures,
the build labels the corresponding transition positions and stores a simple
linear temperature coefficient in `lines.dnu_dt_khz_per_c` and
`lines.dnu_dt_original`.

Landolt method labels are documented in
`docs/build-and-review.md`. Source temperature tokens such as `RT`, `RTemp`, and
`R.Temp` mean room temperature; they are preserved as source text and do not
imply an exact numeric temperature unless one is explicitly available.

## Rebuilding

From the repository root:

```powershell
python NQRDatabase/scripts/build_database.py
```

This regenerates the SQLite database and JSONL exports from the local reference
material and the latest Landolt review decisions.

For build details, review workflow notes, Landolt method labels, and promotion
semantics, see `docs/build-and-review.md`.
