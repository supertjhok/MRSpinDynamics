# NQR Database

This workspace builds a combined nuclear quadrupole resonance database from the
local source material under `References/NQR Data`. It is intended to serve two
audiences at once:

- machine readers, through normalized JSONL exports and an SQLite database;
- human readers, through a review interface and future search/browse UI.

The database currently combines CWRU/UF web captures, the compact NQR database
PDF, Navy/NRL tabulations, King's College notes, and reviewed Landolt-Bornstein
nitrogen-table excerpts.

## What Is Included

The canonical model is organized around compounds, samples, sites, spectral
lines, literature references, and source records. Exported records preserve
source text where possible, while also adding normalized fields such as
`conventional_formula`, `frequency_khz`, `qcc_khz`, isotope labels, and linked
citations.

Generated artifacts live in:

- `data/exports/nqr.sqlite` - relational SQLite database.
- `data/normalized/*.jsonl` - normalized table exports.
- `data/normalized/line_records.jsonl` - denormalized, AI-friendly line records.

Landolt rows that passed human review are promoted into the canonical tables.
Their frequency lists and Q.C.C./eta lists remain deliberately unassigned to
each other unless the source explicitly establishes an assignment.

## Build

From the repository root:

```powershell
python NQRDatabase/scripts/build_database.py
```

This regenerates the SQLite database and JSONL exports from the local reference
material and the accepted Landolt review decisions.

## Review UI

The Landolt review interface can be started with:

```powershell
python NQRDatabase/app/review_server.py
```

Then open `http://127.0.0.1:8765`. Review decisions are stored in
`data/review/landolt_review_decisions.jsonl` and replayed by the build script.

## Documentation

Detailed notes on source coverage, generated tables, Landolt method labels,
review semantics, and the promoted Landolt data model are in
`docs/build-and-review.md`.
