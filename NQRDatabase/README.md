# NQR Spectra Database

This folder contains a curated nuclear quadrupole resonance (NQR) spectra
database. NQR is a radio-frequency spectroscopy method for nuclei with electric
quadrupole moments, such as nitrogen-14, chlorine-35, chlorine-37, bromine-79,
and iodine-127. The most useful raw facts are resonance frequencies, sample
conditions, isotope/site assignments, quadrupole coupling constants, asymmetry
parameters, and the paper or dataset where each value came from.

There is no single standard online NQR spectra database. This project combines
the local source material in `References/NQR Data` into reproducible machine-
readable exports and a small review interface for checking OCR-derived data.

## Current Contents

The current generated database contains:

- 139 compounds
- 171 samples or measurement conditions
- 400 isotope/site records
- 659 NQR line-frequency records
- 95 literature-reference records
- 920 links from compounds, sites, and lines to references

The most convenient files for reuse are:

- `data/exports/nqr.sqlite` - SQLite database with normalized tables.
- `data/normalized/*.jsonl` - one JSON Lines file per normalized table.
- `data/normalized/line_records.jsonl` - denormalized line records for AI tools,
  search indexes, and lightweight applications.

## Data Sources

Every source imported by the build is represented in the `sources` table and
`data/normalized/sources.jsonl`. The main source citations used by this build
are:

- Earlier online NQR database associated with Case Western Reserve University
  and the University of Florida. The local archive contains 25 saved Google
  Sites pages captured on 2020-10-11 under
  `References/NQR Data/CWRU NQR Database/`, plus a compact PDF export,
  `References/NQR Data/NQR Database.pdf`. These pages supply 120 current line
  records in the generated database.
- U.S. Navy / Naval Research Laboratory, `NQR_Data_Tables`. The local copies are
  stored as
  `References/NQR Data/NQRdatabase/NQR_Data_Tables.chm` and PDF exports in
  `References/NQR Data/NQRdatabase/nqr_tables/`. The imported tabulations are
  `NQR_data_tables_summary.pdf`, `NQR_data_tables_summary2.pdf`, and
  `NQR_data_tables_all.pdf`; they supply 77 current line records and compound-
  level citation notes.
- King's College experimental NQR notes in
  `References/NQR Data/NQRdatabase/kings_college_database/`: `Melamine 14N
  NQR.pdf`, `Melamine 14N NQR - update.pdf`, `Metformin HCL 14N NQR.pdf`,
  `Paracetamol 14N NQR.pdf`, and `Population Transfer in a single-axis
  coil.pdf`. These notes supply 25 current line records and preserve acquisition
  or method notes where available.
- H. Chihara and N. Nakamura, *Nuclear Quadrupole Resonance Spectroscopy Data*,
  Landolt-Bornstein, Condensed Matter series, edited by K.-H. Hellwege and
  A. M. Hellwege. The local excerpts are under
  `References/NQR Data/nqr_data/`; imported PDFs include material from
  Condensed Matter III/31A (1993) and III/20A (1988), transition-frequency
  formula pages, nitrogen tables, and nitrogen reference-code pages. The
  reviewed nitrogen-table rows supply 437 current line records.

Individual paper citations from the Navy/NRL and Landolt sources are stored in
`literature_references` and linked through `reference_links`. The compact source
citations above describe the local source collections; use the reference-link
tables when citing a specific measured line or compound.

## Data Model

The canonical database is organized around:

- `compounds` - names, formulas, and display-oriented conventional formulas.
- `samples` - the measured material or condition, including temperature when
  known.
- `sites` - isotope/site information, quadrupole coupling constants, and eta
  values.
- `lines` - resonance frequencies and line-level experimental fields.
- `literature_references` and `reference_links` - provenance for compounds,
  sites, and lines.
- `sources` - the local files or source collections used by the importer.

Frequencies are stored in kHz in canonical fields. Source values and source
formatting are retained in `*_original` fields and JSON `original_record`
payloads.

## Landolt Review Semantics

The Landolt-Bornstein PDFs are layout- and OCR-derived, so their nitrogen-table
entries were reviewed before promotion into the canonical tables. Accepted
review decisions are stored in `data/review/landolt_review_decisions.jsonl` and
replayed during the build.

Landolt rows often report two independent lists for a measurement condition:
line frequencies and Q.C.C./eta pairs. The database deliberately does not infer
which frequency belongs to which Q.C.C./eta pair. In the canonical tables, each
accepted Landolt measurement set becomes one sample; frequencies are stored as
lines under an `unassigned_frequency_list` site, and each Q.C.C./eta pair is
stored as a separate site with `assignment_confidence` set to
`source_reported_unassigned_to_lines`.

## Build

From the repository root:

```powershell
python NQRDatabase/scripts/build_database.py
```

This regenerates the SQLite database and JSONL exports from the local reference
material and accepted Landolt review decisions.

## Review UI

Start the local Landolt review interface with:

```powershell
python NQRDatabase/app/review_server.py
```

Then open `http://127.0.0.1:8765`. The interface displays source-image crops,
parsed fields, measurement sets, frequency lists, and Q.C.C./eta lists.

## Explorer UI

Start the human-facing database explorer with:

```powershell
python NQRDatabase/app/explorer_server.py
```

Then open `http://127.0.0.1:8766`. The explorer searches the canonical
compound, sample, site, line, source, and reference tables. It displays compound
metadata, frequency-line plots, measurement tables, source files, and linked
references.

When a compound has a recognizable CAS alias or name, the explorer attempts to
load a 2D structure image from PubChem in the browser. If no image is available,
or if the browser is offline, it falls back to the stored conventional formula.

## More Detail

Build internals, staged Landolt tables, method labels, and review workflow notes
are documented in `docs/build-and-review.md`.
