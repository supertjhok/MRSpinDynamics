# NQRDatabase — Data License and Provenance

This subproject contains **two kinds of material with two different licenses**:

1. **Code** — the database builders, schema, validation scripts, and the explorer
   app (`app/`, `scripts/`, `schema/`) are licensed under **GPL-3.0-or-later**,
   the same as the rest of the repository. See [`LICENSE`](LICENSE).

2. **Curated data** — the curated NQR database under `data/` (the SQLite export
   `data/exports/nqr.sqlite`, the normalized `data/normalized/*.jsonl` tables, and
   the database schema, normalization, provenance links, derived fields, and the
   selection and arrangement of records) is licensed under the
   **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**.
   To view a copy of this license, visit
   <https://creativecommons.org/licenses/by/4.0/> or see the legal code at
   <https://creativecommons.org/licenses/by/4.0/legalcode>.

## What CC-BY-4.0 covers here

Individual physical measurements (NQR frequencies, quadrupole coupling constants
`QCC`/`C_Q`, asymmetry parameters `eta`, temperatures, etc.) are **facts**, and
facts are not subject to copyright. CC-BY-4.0 is applied to the **curation and
compilation** — the original selection, structuring, normalization, cross-linking,
provenance annotation, and derived/computed fields produced for this database. No
copyright is asserted over the underlying factual values themselves.

## How to attribute

When you use this database, please:

1. Cite the workspace using the repository's
   [`CITATION.cff`](../CITATION.cff) and Zenodo DOI
   ([10.5281/zenodo.21016178](https://doi.org/10.5281/zenodo.21016178)); and
2. Cite the **original measurement sources** for any values you rely on. Each
   record carries its provenance — see the `literature_references` and
   `reference_links` tables (and the `sources` / `landolt_*` tables) — so the
   primary source can always be cited directly.

## Third-party sources (NOT relicensed by this database)

The factual values were compiled from third-party sources. This database stores
extracted **factual values together with citations** to the originals; it does
**not** redistribute, relicense, or reproduce the original copyrighted documents,
text, typesetting, or table layouts. Anyone needing the original materials must
obtain them from the rights holders.

- **Landolt-Börnstein** (H. Chihara and N. Nakamura, *Nuclear Quadrupole
  Resonance Spectroscopy Data*, Condensed Matter series) — © Springer-Verlag.
  Only extracted factual values and citations are stored here.
- **U.S. Navy / Naval Research Laboratory NQR Data Tables** — government-produced
  compilation; factual values are stored with attribution.
- **King's College experimental notes** (melamine, metformin HCl, paracetamol,
  and a population-transfer method note) — used as factual sources with
  attribution.
- **Earlier online NQR database** (Case Western Reserve University / University of
  Florida), captured locally as source material — factual values with attribution.

## Crystal-structure (CIF) files are out of scope

Crystal-structure files under `../QuadrupolarDFT/structures/` are **not part of
this database and are not covered by CC-BY-4.0**. They originate from third-party
crystallographic databases under their own terms (e.g. ICSD / FIZ Karlsruhe and
CCDC / Cambridge Structural Database, which are proprietary and restrict
redistribution; or open sources such as the Crystallography Open Database and IUCr
journals). See the warning in
[`../QuadrupolarDFT/structures/README.md`](../QuadrupolarDFT/structures/README.md)
before reusing any structure file.

## Disclaimer

The data is provided "as is", without warranty of any kind. Values may contain
transcription or OCR errors (the database includes a review workflow and
simulator-based consistency flags precisely because some do). Always verify
against the cited primary source before relying on a value.
