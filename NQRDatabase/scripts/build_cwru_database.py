"""Build the initial NQR database from local NQR reference material.

The importer is intentionally conservative. It normalizes values that are
obvious, keeps the original row text, and records source provenance for every
site and line.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "NQRDatabase"
SCHEMA_SQL = PROJECT / "schema" / "schema.sql"
DEFAULT_CWRU_DIR = ROOT / "References" / "NQR Data" / "CWRU NQR Database"
DEFAULT_NRL_DIR = ROOT / "References" / "NQR Data" / "NQRdatabase" / "nqr_tables"
DEFAULT_KCL_DIR = ROOT / "References" / "NQR Data" / "NQRdatabase" / "kings_college_database"
DEFAULT_LANDOLT_DIR = ROOT / "References" / "NQR Data" / "nqr_data"
DEFAULT_OUTPUT = PROJECT / "data" / "exports" / "nqr.sqlite"
NORMALIZED_DIR = PROJECT / "data" / "normalized"
REVIEW_DIR = PROJECT / "data" / "review"
LANDOLT_CROP_DIR = REVIEW_DIR / "landolt_crops"


FORMULAS = {
    "4-methylimidazole": "C4H6N2",
    "Amlodipine": "C20H25ClN2O5",
    "Ammonium nitrate": "(NH4)(NO3)",
    "Ampicillin trihydrate": "C16H25N3O7S",
    "Benzocaine": "C9H11NO2",
    "Caffeine": "C8H10N4O2",
    "Carbamazepine - Tegretol - Prescription": "C15H12N2O",
    "Chloroquine diphosphate": "C18H32ClN3O8P2",
    "Cladribine - Prescription": "C10H12ClN5O3",
    "Dipicolinic acid": "C7H5NO4",
    "Famotidine - Pepcid - Prescription": "C8H15N7O2S3",
    "Furosemide - Lasix - Prescription": "C12H11ClN2O5S",
    "Glycine": "C2H5NO2",
    "Hydrochlorothiazide": "C7H8ClN3O4S2",
    "L-Histidine": "C6H9N3O2",
    "Metakelfin": "C23H25ClN8O3S",
    "Metformin HCl": "C4H12ClN5",
    "Nicotinamide": "C6H6N2O",
    "Piroxicam - Feldene - prescription": "C15H13N3O4S",
    "Sodium Nitrite": "NaNO2",
    "Sodium chlorate - Cl": "ClNaO3",
    "CL-20": "C6H6N12O12",
    "Cocaine": "C17H21NO4",
    "Cocaine HCl": "C17H21NO4.HCl",
    "Diglycine HCl": "C4H10N2O4.HCl",
    "Dimethyl Urea": "C3H8N2O",
    "Heroin": "C21H23NO5",
    "Heroin HCl": "C21H23NO5.HCl",
    "HMX": "C4H8N8O8",
    "L-Asparagine Monohydrate": "C4H8N2O3.H2O",
    "L-Proline": "C5H9NO2",
    "Melamine": "C3H6N6",
    "Paracetamol": "C8H9NO2",
    "PETN": "C5H8N4O12",
    "Potassium Nitrate": "KNO3",
    "RDX": "C3H6N6O6",
    "Tetryl": "C7H5N5O8",
    "TNT (monoclinic)": "C7H5N3O6",
    "Urea Nitrate": "CH5N3O4",
}

CONVENTIONAL_FORMULAS = {
    **FORMULAS,
    "Ammonium nitrate": "NH4NO3",
    "Chloroquine diphosphate": "C18H32ClN3O8P2",
    "Cocaine HCl": "C17H21NO4.HCl",
    "Diglycine HCl": "C4H10N2O4.HCl",
    "Heroin HCl": "C21H23NO5.HCl",
    "L-Asparagine Monohydrate": "C4H8N2O3.H2O",
    "Potassium Nitrate": "KNO3",
    "Sodium Nitrite": "NaNO2",
    "Sodium chlorate - Cl": "NaClO3",
    "TNT (monoclinic)": "C7H5N3O6",
    "Urea Nitrate": "CH5N3O4",
}

CATEGORIES = {
    "Ammonium nitrate": "explosives",
    "Caffeine": "supplements",
    "Nicotinamide": "supplements",
    "L-Histidine": "supplements",
    "CL-20": "explosives",
    "HMX": "explosives",
    "PETN": "explosives",
    "Potassium Nitrate": "explosives",
    "RDX": "explosives",
    "Tetryl": "explosives",
    "TNT (monoclinic)": "explosives",
    "Urea Nitrate": "explosives",
    "Cocaine": "narcotics",
    "Cocaine HCl": "narcotics",
    "Heroin": "narcotics",
    "Heroin HCl": "narcotics",
    "L-Asparagine Monohydrate": "other compounds",
    "Diglycine HCl": "other compounds",
    "Dimethyl Urea": "other compounds",
    "L-Proline": "other compounds",
    "Melamine": "kcl experimental notes",
    "Metformin HCl": "kcl experimental notes",
    "Paracetamol": "kcl experimental notes",
}

PDF_ONLY_ROWS = [
    {
        "compound": "Glycine",
        "category": "pharmaceutics",
        "headers": ["Type", "Weight-%", "QCC", "eta", "Frequency", "T1", "T2", "Fwhm"],
        "rows": [
            ["amine-14N", "18.7", "1193", "0.528", "1052", "43.4", "12.5", "2.8"],
            ["", "", "", "", "737", "50", "17.1", "0.8"],
        ],
        "source_page": "NQR Database.pdf page 1",
    },
    {
        "compound": "Nicotinamide",
        "category": "supplements",
        "headers": ["Frequency"],
        "rows": [["2307"]],
        "source_page": "NQR Database.pdf page 7",
    },
]

SKIP_PAGES = {"NQR database", "Explosives", "Pharmaceutics", "Supplements"}

LANDOLT_PDFS = {
    "landolt_front_matter_pdf": "b43999.pdf",
    "landolt_transition_frequencies_pdf": "10333382_11.pdf",
    "landolt_transition_frequencies_alt_pdf": "10044098_11.pdf",
    "landolt_intro_pdf": "10333382_15.pdf",
    "landolt_nitrogen_table_a_pdf": "10044098_28.pdf",
    "landolt_nitrogen_table_b_pdf": "10333382_42.pdf",
    "landolt_nitrogen_references_pdf": "10333382_43.pdf",
}
LANDOLT_REVIEW_DECISIONS = PROJECT / "data" / "review" / "landolt_review_decisions.jsonl"

LANDOLT_COLUMN_DEFINITIONS = [
    ("subst_no", "Subst. No.", "Substances are numbered sequentially within a table; index entries combine table number and substance number.", "p. 80"),
    ("formula", "Formula", "Gross molecular formula is given. The same substance can appear in multiple tables when different elements are measured.", "p. 80"),
    ("nucleus", "Nucl.", "Nucleus at resonance, such as N-14 or Cl-35.", "p. 80"),
    ("method", "Meth.", "Experimental or computational method code reported by Landolt-Bornstein.", "p. 80"),
    ("temperature", "Temp.", "Temperature of the measurement, usually in kelvin; RT indicates room temperature.", "p. 80"),
    ("frequency", "Fres.", "Nuclear quadrupole resonance frequency values, typically in MHz in the Landolt tables.", "p. 80"),
    ("qcc", "Q.C.C.", "Quadrupole coupling constant. The Q.C.C. and eta values correspond to each other when reported in the same line.", "p. 81"),
    ("eta", "eta", "Asymmetry parameter of the quadrupole coupling tensor.", "p. 81"),
    ("remarks", "Rem.", "Asterisks indicate remarks in the substance footnote.", "p. 81"),
    ("reference", "Ref.", "Reference code for the table; blank entries indicate the same reference as the preceding line.", "p. 81"),
]

LANDOLT_METHOD_DEFINITIONS = {
    "C": "Continuous wave method",
    "D": "Double resonance method",
    "P": "Pulse method",
    "M": "NMR method",
    "E": "Other methods",
    "X": "Method not described in the original paper or not recorded in the database at the early stage",
}
LANDOLT_ROW_NUMBER_PATTERN = r"^[\s'`\u2018\u2019]*(?P<num>\d{2,3})\.?\s+(?P<after_number>.+)$"
LANDOLT_FOOTNOTE_NUMBER_PATTERN = r"^[\s'`\u2018\u2019]*(?P<num>\d{2,3})\.\.?\s+(?P<text>.+)$"

LANDOLT_TRANSITION_EQUATIONS = [
    {
        "id": "landolt_transition_i1_nu_plus",
        "nuclear_spin": "1",
        "transition_label": "nu+",
        "expression_text": "nu+ = (3 e^2 q Q / 4 h) * (1 + eta / 3)",
        "expression_latex": r"\nu_+ = \frac{3 e^2 q Q}{4h}\left(1+\frac{\eta}{3}\right)",
        "confidence": "curated",
        "notes": "Spin-1 transition-frequency formula from Landolt-Bornstein section 2.7.",
    },
    {
        "id": "landolt_transition_i1_nu_minus",
        "nuclear_spin": "1",
        "transition_label": "nu-",
        "expression_text": "nu- = (3 e^2 q Q / 4 h) * (1 - eta / 3)",
        "expression_latex": r"\nu_- = \frac{3 e^2 q Q}{4h}\left(1-\frac{\eta}{3}\right)",
        "confidence": "curated",
        "notes": "Spin-1 transition-frequency formula from Landolt-Bornstein section 2.7.",
    },
    {
        "id": "landolt_transition_i1_nu_zero",
        "nuclear_spin": "1",
        "transition_label": "nu0",
        "expression_text": "nu0 = (e^2 q Q / 2 h) * eta",
        "expression_latex": r"\nu_0 = \frac{e^2 q Q}{2h}\eta",
        "confidence": "curated",
        "notes": "Spin-1 zero-frequency difference relation from Landolt-Bornstein section 2.7.",
    },
    {
        "id": "landolt_transition_i3_2",
        "nuclear_spin": "3/2",
        "transition_label": "1/2 <-> 3/2",
        "expression_text": "nu = (e^2 q Q / 2 h) * sqrt(1 + eta^2 / 3)",
        "expression_latex": r"\nu = \frac{e^2 q Q}{2h}\sqrt{1+\frac{\eta^2}{3}}",
        "confidence": "curated",
        "notes": "Spin-3/2 transition-frequency formula from Landolt-Bornstein section 2.7.",
    },
    {
        "id": "landolt_transition_higher_spin_series_raw",
        "nuclear_spin": "5/2, 7/2, 9/2",
        "transition_label": "series expansions",
        "expression_text": "Higher-spin transition-frequency series are present in the source table, but OCR is unreliable; inspect source page before using numerically.",
        "expression_latex": None,
        "confidence": "raw_source_notice",
        "notes": "The page contains eta-series expansions for I=5/2, 7/2, and 9/2. They should be transcribed from page image/manual review before computational use.",
    },
]

NRL_REFERENCE_ROWS = [
    ("Ammonium nitrate", "1", "J. Seliger, V. Zagar and R. Blinc, Zeitschrift für Physik B, Condensed Matter and Quanta 25, 189 (1976).", "p. 3", "literature"),
    ("CL-20", "1", "NRL data taken at NSWC Indian Head, October 2002, for epsilon CL-20 from Thiokol.", "p. 5", "source_note"),
    ("CL-20", "2", "Temperature coefficient between 21 C and 25 C (NRL data).", "p. 5", "source_note"),
    ("HMX", "1", "Site numbering and ring-14N (1=axial and 2=equatorial) NQR line assignments from R. A. Landers, T. B. Brill and R. A. Marino, J. Phys. Chem. 85, 2618 (1981).", "p. 6", "literature"),
    ("HMX", "2", "Ring-14N NQR frequencies from reference 1.", "p. 6", "source_note"),
    ("HMX", "3", "Temperature coefficients between -10 C and 42 C from reference 1.", "p. 6", "source_note"),
    ("HMX", "4", "M. L. Buess, J. P. Yesinowski and A. N. Garroway, \"Anomalous 14N NQR Steady State Free Precession Responses\", poster presented at the 34th Experimental Nuclear Magnetic Resonance Conference (ENC), March 14-18, 1993, St. Louis, MO.", "p. 6", "conference_poster"),
    ("PETN", "1", "Temperature coefficients between room temperature (NRL data) and 77 K data from R. A. Marino, \"Nitrogen-14 and Nitrogen-15 Wide Line NMR Spectroscopic Studies of Nitrocellulose\", Final Technical Report for Task 7-09, Dept. of the Army, Battelle, P. O. Box 12207, RPT, NC 27707 (1987).", "p. 7", "technical_report"),
    ("PETN", "2", "Temperature coefficients between -50 C and +50 C (NRL data taken at the FAA Technical Center, 16-19 May 2000).", "p. 7", "source_note"),
    ("Potassium Nitrate", "1", "T. J. Barstow, J. Chem. Soc. Faraday Trans. 87(15), 2453 (1991).", "p. 8", "literature"),
    ("RDX", "1", "Site numbering and ring-14N NQR line assignments from R. J. Karpowicz and T. B. Brill, J. Phys. Chem. 87, 2109 (1983).", "p. 10", "literature"),
    ("RDX", "2", "Ring-14N NQR frequencies from reference 1. Nitro-14N NQR frequencies from NRL data.", "p. 10", "source_note"),
    ("RDX", "3", "Temperature coefficients between -11 C and 40 C from reference 1 unless otherwise noted.", "p. 10", "source_note"),
    ("RDX", "4", "Temperature coefficient from NRL data at 22 C, 23 C and 24 C.", "p. 10", "source_note"),
    ("Tetryl", "1", "NRL data taken at the FAA Technical Center, 10-14 January 2000, for a tetryl formulation supplied by NSWC, Indian Head, MD. Numbers in parenthesis are for a tetryl formulation supplied by Explotech.", "p. 11", "source_note"),
    ("Tetryl", "2", "Temperature coefficients between -50 C and +50 C (NRL data taken at the FAA Technical Center, 16-19 May 2000).", "p. 11", "source_note"),
    ("Tetryl", "3", "Relative to the echo intensity at the start of the SLSE echo train with a 1.0 ms pulse interval.", "p. 11", "source_note"),
    ("TNT (monoclinic)", "1", "NQR line assignments and temperature coefficients determined by R. A. Marino and R. F. Connors, Journal of Molecular Structure 111, 323 (1983).", "p. 13", "literature"),
    ("Urea Nitrate", "1", "Temperature coefficients between 77 K and 275 K, H. Negita, T. Kubo and H. Kato, Bull. Chem. Soc. Jpn. 54, 391 (1981).", "p. 15", "literature"),
    ("Cocaine", "1", "J. P. Yesinowski, M. L. Buess, A. N. Garroway, M. Ziegeweid and A. Pines, Analytical Chemistry 67, 2256 (1995).", "p. 16", "literature"),
    ("Cocaine", "2", "Temperature coefficients between 77 K and 295 K from reference 1.", "p. 16", "source_note"),
    ("Cocaine", "3", "M. L. Buess, J. P. Yesinowski and A. N. Garroway, \"Anomalous 14N NQR Steady State Free Precession Responses\", poster presented at the 34th Experimental Nuclear Magnetic Resonance Conference (ENC), March 14-18, 1993, St. Louis, MO.", "p. 16", "conference_poster"),
    ("Cocaine HCl", "1", "J. P. Yesinowski, M. L. Buess, A. N. Garroway, M. Ziegeweid and A. Pines, Analytical Chemistry 67, 2256 (1995).", "p. 17", "literature"),
    ("Cocaine HCl", "2", "Temperature coefficients between 77 K and 295 K from reference 1.", "p. 17", "source_note"),
    ("Heroin HCl", "1", "E. F. Emery, K. McGrath, J. P. Yesinowski, J. B. Miller and L. G. Butler, \"Nitrogen-14 NQR and NMR of Heroin Hydrochloride\", poster presented at the 38th Experimental Nuclear Magnetic Resonance Conference (ENC), March 23-27, 1997, Orlando, FL.", "p. 18", "conference_poster"),
    ("L-Asparagine Monohydrate", "1", "A. Naito and C. A. McDowell, J. Chem. Phys. 81, 11 (1984).", "p. 19", "literature"),
    ("L-Asparagine Monohydrate", "2", "NRL data, 20 April and 4-5 June, 2001. No attempt was made to find the NQR lines of site 1 at 2233 kHz and 1782 kHz (Ref. 1).", "p. 19", "source_note"),
    ("L-Asparagine Monohydrate", "3", "Estimated from 77 K frequencies (M. J. Hunt, J. Magn. Resonance 15, 1 (1974)).", "p. 19", "literature"),
    ("Diglycine HCl", "1", "NRL data, 26 April - 16 May 2001.", "p. 20", "source_note"),
    ("Diglycine HCl", "2", "There are three possible ways to pair the NQR lines; relaxation times suggest pairings 1b: (982 kHz, 815 kHz) and 2b: (834 kHz, 725 kHz).", "p. 20-21", "source_note"),
    ("Dimethyl Urea", "1", "Dinesh and M. T. Rogers, J. Chem. Phys. 57, 9, 3726-3728 (1972).", "p. 21", "literature"),
    ("Dimethyl Urea", "2", "J. Murgich, R. M. Santana and J. Abanero, Magn. Reson. Chem. 23, 3, 145-149 (1985).", "p. 21", "literature"),
    ("Glycine", "1", "NRL data, 9-11 April 2001.", "p. 22", "source_note"),
    ("Glycine", "2", "Estimated from 90 K frequencies (D. T. Edmonds and C. P. Summers, Chem Phys Lett 41, 3 (1976)).", "p. 22", "literature"),
    ("L-Proline", "1", "NRL data, 26 March - 6 April 2001.", "p. 23", "source_note"),
    ("L-Proline", "2", "Estimated from 77 K frequencies (D. T. Edmonds, M. J. Hunt and A. L. McKay, J. Magn. Resonance 9, 1 (1973)).", "p. 23", "literature"),
    ("Sodium Nitrite", "1", "Estimated from temperature dependence data for the coupling constant and asymmetry parameter given by G. Petersen and P. J. Bray, J. Chem. Phys. 64, 2, 522-530 (1976).", "p. 23-24", "literature"),
]


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self.stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            table = {"rows": [], "current_row": None, "current_cell": None}
            self.tables.append(table)
            self.stack.append(table)
        elif tag == "tr" and self.stack:
            self.stack[-1]["current_row"] = []
        elif tag in {"td", "th"} and self.stack:
            self.stack[-1]["current_cell"] = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.stack:
            table = self.stack[-1]
            if table["current_cell"] is not None and table["current_row"] is not None:
                text = normalize_space(" ".join(table["current_cell"]))
                table["current_row"].append(text)
            table["current_cell"] = None
        elif tag == "tr" and self.stack:
            table = self.stack[-1]
            row = table["current_row"]
            if row and any(cell for cell in row):
                table["rows"].append(row)
            table["current_row"] = None
        elif tag == "table" and self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1]["current_cell"] is not None:
            self.stack[-1]["current_cell"].append(data)


@dataclass
class BuildState:
    sources: dict[str, dict[str, Any]]
    compounds: dict[str, dict[str, Any]]
    aliases: set[tuple[str, str]]
    samples: dict[str, dict[str, Any]]
    sites: dict[str, dict[str, Any]]
    lines: dict[str, dict[str, Any]]
    literature_references: dict[str, dict[str, Any]]
    reference_links: dict[str, dict[str, Any]]
    nqr_transition_equations: dict[str, dict[str, Any]]
    landolt_column_definitions: dict[str, dict[str, Any]]
    landolt_page_extracts: dict[str, dict[str, Any]]
    landolt_compound_entries: dict[str, dict[str, Any]]
    landolt_reference_codes: dict[str, dict[str, Any]]
    landolt_measurement_sets: dict[str, dict[str, Any]]
    landolt_frequency_records: dict[str, dict[str, Any]]
    landolt_qcc_eta_records: dict[str, dict[str, Any]]
    landolt_review_queue: dict[str, dict[str, Any]]


def normalize_space(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def first_number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def parse_time_s(value: str | None) -> float | None:
    text = normalize_space(value)
    if not text:
        return None
    number = first_number(text)
    if number is None:
        return None
    lower = text.lower()
    if "ms" in lower:
        return number / 1000.0
    if "us" in lower or "micro" in lower:
        return number / 1_000_000.0
    return number


def parse_temperature_k(value: str | None) -> float | None:
    text = normalize_space(value)
    match = re.search(r"@\s*(\d+(?:\.\d+)?)\s*K", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def title_from_path(path: Path) -> str:
    title = path.stem
    return re.sub(r"\s+-\s+NQR database$", "", title).strip()


def clean_header(header: str) -> str:
    text = normalize_space(header).lower()
    text = text.replace("η", "eta")
    if text == "t2#":
        return "t2_star"
    text = text.replace("e2qq/h (khz)", "qcc")
    text = text.replace("e2qq/h", "qcc")
    text = text.replace("frequency (khz)", "frequency")
    text = text.replace("line width", "line_width")
    text = text.replace("weight %", "weight-%")
    text = text.replace("weight-%", "weight_percent")
    text = text.replace("fwhm (khz)", "fwhm")
    text = text.replace("qcc", "qcc")
    return re.sub(r"[^a-z0-9#]+", "_", text).strip("_")


def table_candidates(path: Path) -> list[dict[str, Any]]:
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    candidates = []
    for table in parser.tables:
        rows = table["rows"]
        for index, row in enumerate(rows):
            cleaned = [clean_header(cell) for cell in row]
            if "type" in cleaned and ("frequency" in cleaned or "qcc" in cleaned):
                candidates.append({"header": row, "rows": rows[index + 1 :]})
                break
    return candidates


def best_candidate(path: Path) -> dict[str, Any] | None:
    candidates = table_candidates(path)
    useful = []
    for candidate in candidates:
        rows = [
            row
            for row in candidate["rows"]
            if row and not all(normalize_space(cell).lower() == "sort" for cell in row)
        ]
        if rows:
            useful.append({"header": candidate["header"], "rows": rows})
    if not useful:
        return None
    return max(useful, key=lambda item: len(item["rows"]))


def category_for(compound: str) -> str:
    return CATEGORIES.get(compound, "pharmaceutics")


def source_id_for_path(path: Path) -> str:
    return "cwru_html_" + slug(path.stem)


def ensure_compound(state: BuildState, name: str, category: str | None = None) -> str:
    compound_id = "compound:" + slug(name)
    state.compounds.setdefault(
        compound_id,
        {
            "id": compound_id,
            "canonical_name": name,
            "formula": FORMULAS.get(name),
            "conventional_formula": CONVENTIONAL_FORMULAS.get(name, FORMULAS.get(name)),
            "category": category or category_for(name),
            "notes": None,
        },
    )
    return compound_id


def ensure_sample(state: BuildState, compound_id: str, compound_name: str) -> str:
    sample_id = "sample:" + slug(compound_name) + ":default"
    state.samples.setdefault(
        sample_id,
        {
            "id": sample_id,
            "compound_id": compound_id,
            "label": "default",
            "form": None,
            "phase": None,
            "temperature_k": None,
            "notes": None,
        },
    )
    return sample_id


def extract_isotope(site_label: str | None, compound: str) -> str | None:
    text = normalize_space(site_label)
    match = re.search(r"\b(\d{1,3}[A-Z][a-z]?)\b", text)
    if match:
        return match.group(1)
    if "chlorate" in compound.lower() or normalize_space(site_label).lower() == "cl":
        return "35Cl"
    if text:
        return "14N"
    return None


def site_key(
    source_id: str,
    sample_id: str,
    site_number: str | None,
    site_label: str | None,
    qcc_khz: float | None,
    eta: float | None,
) -> str:
    parts = [
        source_id,
        sample_id.replace("sample:", ""),
        site_number or "",
        site_label or "unassigned",
        "" if qcc_khz is None else f"{qcc_khz:g}",
        "" if eta is None else f"{eta:g}",
    ]
    return "site:" + slug(":".join(parts))


def add_line_rows(
    state: BuildState,
    compound_name: str,
    source_id: str,
    headers: list[str],
    rows: list[list[str]],
    curation_method: str,
) -> None:
    compound_id = ensure_compound(state, compound_name, category_for(compound_name))
    sample_id = ensure_sample(state, compound_id, compound_name)
    header_keys = [clean_header(header) for header in headers]
    current_site: dict[str, Any] = {
        "site_number": None,
        "site_label": None,
        "weight_percent": None,
        "qcc_khz": None,
        "eta": None,
    }

    for row_number, row in enumerate(rows, start=1):
        if not row or all(not normalize_space(cell) for cell in row):
            continue
        if all(normalize_space(cell).lower() == "sort" for cell in row):
            continue
        padded = row + [""] * max(0, len(header_keys) - len(row))
        values = {key: normalize_space(padded[i]) for i, key in enumerate(header_keys)}

        raw_type = values.get("type") or ""
        if raw_type:
            if raw_type.isdigit():
                current_site["site_number"] = raw_type
            else:
                current_site["site_label"] = raw_type
        qcc = first_number(values.get("qcc"))
        eta = first_number(values.get("eta"))
        weight = first_number(values.get("weight_percent"))
        if qcc is not None:
            current_site["qcc_khz"] = qcc
        if eta is not None:
            current_site["eta"] = eta
        if weight is not None:
            current_site["weight_percent"] = weight

        frequency_original = values.get("frequency") or values.get("line_type") or ""
        frequency = first_number(frequency_original)
        temperature_k = parse_temperature_k(frequency_original)
        site_label = current_site["site_label"]
        if not site_label and current_site["site_number"]:
            site_label = current_site["site_number"]

        site_id = site_key(
            source_id,
            sample_id,
            current_site["site_number"],
            site_label,
            current_site["qcc_khz"],
            current_site["eta"],
        )
        original = {
            "headers": headers,
            "row": row,
            "row_number": row_number,
            "curation_method": curation_method,
        }
        state.sites.setdefault(
            site_id,
            {
                "id": site_id,
                "sample_id": sample_id,
                "site_number": current_site["site_number"],
                "isotope": extract_isotope(site_label, compound_name),
                "site_label": site_label,
                "weight_percent": current_site["weight_percent"],
                "qcc_khz": current_site["qcc_khz"],
                "eta": current_site["eta"],
                "assignment_confidence": "source_reported" if site_label else "unassigned",
                "source_id": source_id,
                "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
            },
        )

        if frequency is None:
            continue

        form = values.get("form") or None
        line_width_original = values.get("line_width") or values.get("fwhm") or None
        line_width = first_number(line_width_original)
        line_id = "line:" + slug(
            ":".join(
                [
                    site_id.replace("site:", ""),
                    source_id,
                    frequency_original or str(frequency),
                    form or "",
                    str(row_number),
                ]
            )
        )
        state.lines[line_id] = {
            "id": line_id,
            "site_id": site_id,
            "frequency_khz": frequency,
            "frequency_original": frequency_original,
            "transition_label": None,
            "fwhm_khz": first_number(values.get("fwhm")),
            "line_width_khz": line_width,
            "line_width_original": line_width_original,
            "t1_s": parse_time_s(values.get("t1")),
            "t1_original": values.get("t1") or None,
            "t2_s": parse_time_s(values.get("t2")),
            "t2_original": values.get("t2") or None,
            "t2_star_s": parse_time_s(values.get("t2_star")),
            "t2_star_original": values.get("t2_star") or None,
            "dnu_dt_khz_per_c": None,
            "dnu_dt_original": None,
            "polarization_factor": None,
            "polarization_factor_original": None,
            "temperature_k": temperature_k,
            "form": form,
            "source_id": source_id,
            "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
        }


def build_state(cwru_dir: Path) -> BuildState:
    state = BuildState({}, {}, set(), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for path in sorted(cwru_dir.glob("*.html")):
        title = title_from_path(path)
        source_id = source_id_for_path(path)
        state.sources[source_id] = {
            "id": source_id,
            "title": title,
            "source_type": "cwru_google_sites_wayback_html",
            "relative_path": str(path.relative_to(ROOT)),
            "url": None,
            "captured_at": "2020-10-11",
            "notes": "Saved Google Sites page from the CWRU/UF NQR database snapshot.",
        }
        if title in SKIP_PAGES:
            continue
        candidate = best_candidate(path)
        if candidate:
            add_line_rows(
                state,
                title,
                source_id,
                candidate["header"],
                candidate["rows"],
                "parsed_html_table",
            )

    pdf_source_id = "cwru_compact_pdf"
    state.sources[pdf_source_id] = {
        "id": pdf_source_id,
        "title": "NQR Database.pdf",
        "source_type": "cwru_compact_pdf",
        "relative_path": str((ROOT / "References" / "NQR Data" / "NQR Database.pdf").relative_to(ROOT)),
        "url": None,
        "captured_at": None,
        "notes": "Compact PDF export of the CWRU/UF NQR database. Used here for entries missing from the local HTML snapshot.",
    }
    for block in PDF_ONLY_ROWS:
        if not has_lines_for_compound(state, block["compound"]):
            add_line_rows(
                state,
                block["compound"],
                pdf_source_id,
                block["headers"],
                block["rows"],
                "manual_from_pdf_text:" + block["source_page"],
            )
    return state


def has_lines_for_compound(state: BuildState, compound_name: str) -> bool:
    compound_id = "compound:" + slug(compound_name)
    sample_ids = {sample["id"] for sample in state.samples.values() if sample["compound_id"] == compound_id}
    site_ids = {site["id"] for site in state.sites.values() if site["sample_id"] in sample_ids}
    return any(line["site_id"] in site_ids for line in state.lines.values())


def split_lines(value: str | None) -> list[str]:
    text = normalize_space(value).replace(" \n", "\n") if value else ""
    if "\n" in (value or ""):
        return [normalize_space(part) for part in (value or "").splitlines() if normalize_space(part)]
    return [text] if text else []


def split_nrl_type_cell(value: str | None) -> list[str]:
    parts = split_lines(value)
    merged: list[str] = []
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if part in {"14", "35", "39"} and idx + 1 < len(parts):
            merged.append(f"{part} {parts[idx + 1]}")
            idx += 2
        else:
            merged.append(part)
            idx += 1
    return merged


def dash_to_none(value: str | None) -> str | None:
    text = normalize_space(value)
    if not text or text in {"--", "None"}:
        return None
    return text


def normalize_compound_name(value: str | None) -> str:
    text = normalize_space(value).replace("\n", " ")
    replacements = {
        "Ammonium Nitrate": "Ammonium nitrate",
        "Potassium Nitrate": "Potassium Nitrate",
        "Sodium Nitrite": "Sodium Nitrite",
        "Dimethyl Urea": "Dimethyl Urea",
        "TNT (monoclinic)": "TNT (monoclinic)",
    }
    return replacements.get(text, text)


def normalize_nrl_type(value: str | None, previous: str | None = None) -> str | None:
    text = normalize_space((value or "").replace("\n", " "))
    if text == '"':
        return previous
    if not text:
        return previous
    text = re.sub(r"\b14\s+([A-Za-z]+)-\s*N\b", r"\1-14N", text)
    text = re.sub(r"\b35\s+Cl\b", "35Cl", text)
    text = re.sub(r"\b39\s+K\b", "39K", text)
    return text


def add_source(
    state: BuildState,
    source_id: str,
    title: str,
    source_type: str,
    path: Path | None,
    notes: str | None = None,
) -> None:
    state.sources[source_id] = {
        "id": source_id,
        "title": title,
        "source_type": source_type,
        "relative_path": str(path.relative_to(ROOT)) if path else None,
        "url": None,
        "captured_at": None,
        "notes": notes,
    }


def extract_year(text: str) -> int | None:
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return int(matches[-1]) if matches else None


def add_literature_reference(
    state: BuildState,
    compound_name: str,
    source_id: str,
    source_reference_number: str,
    citation_text: str,
    source_page: str,
    reference_type: str,
) -> str:
    reference_id = "ref:" + slug(":".join([source_id, compound_name, source_reference_number, citation_text[:80]]))
    state.literature_references.setdefault(
        reference_id,
        {
            "id": reference_id,
            "citation_text": citation_text,
            "reference_type": reference_type,
            "authors": None,
            "year": extract_year(citation_text),
            "title": None,
            "journal": None,
            "doi": None,
            "source_id": source_id,
            "source_page": source_page,
            "source_reference_number": source_reference_number,
            "original_text": citation_text,
        },
    )
    return reference_id


def add_reference_link(
    state: BuildState,
    reference_id: str,
    source_id: str,
    link_type: str,
    compound_id: str | None = None,
    site_id: str | None = None,
    line_id: str | None = None,
    note: str | None = None,
) -> None:
    link_id = "reflink:" + slug(":".join([reference_id, compound_id or "", site_id or "", line_id or "", link_type]))
    state.reference_links.setdefault(
        link_id,
        {
            "id": link_id,
            "reference_id": reference_id,
            "compound_id": compound_id,
            "site_id": site_id,
            "line_id": line_id,
            "source_id": source_id,
            "link_type": link_type,
            "note": note,
        },
    )


def upsert_site(
    state: BuildState,
    compound_name: str,
    source_id: str,
    site_number: str | None,
    site_label: str | None,
    isotope: str | None,
    weight_percent: float | None,
    qcc_khz: float | None,
    eta: float | None,
    original: dict[str, Any],
) -> str:
    compound_id = ensure_compound(state, compound_name, category_for(compound_name))
    sample_id = ensure_sample(state, compound_id, compound_name)
    site_id = site_key(source_id, sample_id, site_number, site_label, qcc_khz, eta)
    state.sites.setdefault(
        site_id,
        {
            "id": site_id,
            "sample_id": sample_id,
            "site_number": site_number,
            "isotope": isotope or extract_isotope(site_label, compound_name),
            "site_label": site_label,
            "weight_percent": weight_percent,
            "qcc_khz": qcc_khz,
            "eta": eta,
            "assignment_confidence": "source_reported" if site_number or site_label else "unassigned",
            "source_id": source_id,
            "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
        },
    )
    return site_id


def add_line_record(
    state: BuildState,
    source_id: str,
    site_id: str,
    frequency_original: str,
    original: dict[str, Any],
    row_index: int,
    transition_label: str | None = None,
    fwhm_original: str | None = None,
    t1_original: str | None = None,
    t2_original: str | None = None,
    dnu_dt_original: str | None = None,
    polarization_factor_original: str | None = None,
    form: str | None = None,
    temperature_k: float | None = None,
) -> str | None:
    frequency = first_number(frequency_original)
    if frequency is None:
        return None
    line_id = "line:" + slug(
        ":".join(
            [
                site_id.replace("site:", ""),
                source_id,
                transition_label or "",
                frequency_original,
                form or "",
                str(row_index),
            ]
        )
    )
    state.lines[line_id] = {
        "id": line_id,
        "site_id": site_id,
        "frequency_khz": frequency,
        "frequency_original": frequency_original,
        "transition_label": transition_label,
        "fwhm_khz": first_number(fwhm_original),
        "line_width_khz": first_number(fwhm_original),
        "line_width_original": dash_to_none(fwhm_original),
        "t1_s": parse_time_s(t1_original),
        "t1_original": dash_to_none(t1_original),
        "t2_s": parse_time_s(t2_original),
        "t2_original": dash_to_none(t2_original),
        "t2_star_s": None,
        "t2_star_original": None,
        "dnu_dt_khz_per_c": first_number(dnu_dt_original),
        "dnu_dt_original": dash_to_none(dnu_dt_original),
        "polarization_factor": first_number(polarization_factor_original),
        "polarization_factor_original": dash_to_none(polarization_factor_original),
        "temperature_k": temperature_k or parse_temperature_k(frequency_original),
        "form": form,
        "source_id": source_id,
        "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }
    return line_id


def import_nrl_summaries(state: BuildState, nrl_dir: Path) -> None:
    site_pdf = nrl_dir / "NQR_data_tables_summary2.pdf"
    line_pdf = nrl_dir / "NQR_data_tables_summary.pdf"
    add_source(
        state,
        "nrl_detailed_tables_pdf",
        "NQR_data_tables_all.pdf",
        "nrl_nqr_data_tables_detailed_pdf",
        nrl_dir / "NQR_data_tables_all.pdf",
        "Detailed NRL/Navy NQR Data Tables export containing compound-level citations and notes.",
    )
    add_source(
        state,
        "nrl_site_summary_pdf",
        "NQR_data_tables_summary2.pdf",
        "nrl_nqr_data_tables_site_summary_pdf",
        site_pdf,
        "Site summary from the NRL/Navy NQR Data Tables CHM/PDF export.",
    )
    add_source(
        state,
        "nrl_line_summary_pdf",
        "NQR_data_tables_summary.pdf",
        "nrl_nqr_data_tables_line_summary_pdf",
        line_pdf,
        "Line summary from the NRL/Navy NQR Data Tables CHM/PDF export.",
    )
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pdfplumber is required for NRL PDF table extraction") from exc

    nrl_site_index: dict[tuple[str, str], str] = {}
    with pdfplumber.open(site_pdf) as pdf:
        previous_by_compound: dict[str, dict[str, str | None]] = {}
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_number, table in enumerate(page.extract_tables(), start=1):
                for row_number, row in enumerate(table, start=1):
                    if row and normalize_space(row[0]).lower() == "substance":
                        continue
                    if len(row) < 6:
                        continue
                    compound = normalize_compound_name(row[0])
                    cols = [
                        split_lines(row[1]),
                        split_nrl_type_cell(row[2]),
                        split_lines(row[3]),
                        split_lines(row[4]),
                        split_lines(row[5]),
                    ]
                    count = max(len(col) for col in cols)
                    previous = previous_by_compound.setdefault(
                        compound, {"type": None, "weight": None, "qcc": None, "eta": None}
                    )
                    for idx in range(count):
                        site_number = cols[0][idx] if idx < len(cols[0]) else None
                        site_type = cols[1][idx] if idx < len(cols[1]) else None
                        weight = cols[2][idx] if idx < len(cols[2]) else None
                        qcc = cols[3][idx] if idx < len(cols[3]) else None
                        eta = cols[4][idx] if idx < len(cols[4]) else None
                        site_label = normalize_nrl_type(site_type, previous["type"])
                        if weight == '"':
                            weight = previous["weight"]
                        if qcc == '"':
                            qcc = previous["qcc"]
                        if eta == '"':
                            eta = previous["eta"]
                        previous.update({"type": site_label, "weight": weight, "qcc": qcc, "eta": eta})
                        original = {
                            "curation_method": "pdfplumber_table",
                            "source_pdf": site_pdf.name,
                            "page": page_number,
                            "table": table_number,
                            "row": row,
                            "expanded_index": idx + 1,
                        }
                        site_id = upsert_site(
                            state,
                            compound,
                            "nrl_site_summary_pdf",
                            site_number,
                            site_label,
                            extract_isotope(site_label, compound),
                            first_number(weight),
                            first_number(qcc),
                            first_number(eta),
                            original,
                        )
                        if site_number:
                            nrl_site_index[(compound, site_number)] = site_id

    with pdfplumber.open(line_pdf) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_number, table in enumerate(page.extract_tables(), start=1):
                for row_number, row in enumerate(table, start=1):
                    if row and normalize_space(row[0]).lower() == "substance":
                        continue
                    if len(row) < 7:
                        continue
                    compound = normalize_compound_name(row[0])
                    cols = [split_lines(cell) for cell in row[1:7]]
                    count = max(len(col) for col in cols)
                    for idx in range(count):
                        site_number = cols[0][idx] if idx < len(cols[0]) else None
                        frequency = cols[1][idx] if idx < len(cols[1]) else None
                        fwhm = cols[2][idx] if idx < len(cols[2]) else None
                        t1 = cols[3][idx] if idx < len(cols[3]) else None
                        t2 = cols[4][idx] if idx < len(cols[4]) else None
                        dnu = cols[5][idx] if idx < len(cols[5]) else None
                        if not frequency:
                            continue
                        site_id = nrl_site_index.get((compound, site_number or ""))
                        if not site_id:
                            original_site = {
                                "curation_method": "inferred_from_nrl_line_summary",
                                "source_pdf": line_pdf.name,
                                "page": page_number,
                                "table": table_number,
                                "row": row,
                                "expanded_index": idx + 1,
                            }
                            site_id = upsert_site(
                                state,
                                compound,
                                "nrl_line_summary_pdf",
                                site_number,
                                site_number,
                                None,
                                None,
                                None,
                                None,
                                original_site,
                            )
                        original = {
                            "curation_method": "pdfplumber_table",
                            "source_pdf": line_pdf.name,
                            "page": page_number,
                            "table": table_number,
                            "row": row,
                            "expanded_index": idx + 1,
                        }
                        add_line_record(
                            state,
                            "nrl_line_summary_pdf",
                            site_id,
                            frequency,
                            original,
                            row_index=(page_number * 1000 + table_number * 100 + row_number * 10 + idx),
                            fwhm_original=fwhm,
                            t1_original=t1,
                            t2_original=t2,
                            dnu_dt_original=dnu,
                        )


def import_nrl_references(state: BuildState) -> None:
    for compound_name, ref_number, citation_text, source_page, reference_type in NRL_REFERENCE_ROWS:
        compound_id = ensure_compound(state, compound_name, category_for(compound_name))
        reference_id = add_literature_reference(
            state,
            compound_name,
            "nrl_detailed_tables_pdf",
            ref_number,
            citation_text,
            source_page,
            reference_type,
        )
        add_reference_link(
            state,
            reference_id,
            "nrl_detailed_tables_pdf",
            "compound",
            compound_id=compound_id,
            note=f"Navy detailed-table reference {ref_number} for {compound_name}.",
        )
        sample_ids = {sample["id"] for sample in state.samples.values() if sample["compound_id"] == compound_id}
        site_ids = {site["id"] for site in state.sites.values() if site["sample_id"] in sample_ids}
        for line in state.lines.values():
            if line["site_id"] in site_ids and line["source_id"] == "nrl_line_summary_pdf":
                add_reference_link(
                    state,
                    reference_id,
                    "nrl_detailed_tables_pdf",
                    "line",
                    compound_id=compound_id,
                    site_id=line["site_id"],
                    line_id=line["id"],
                    note=f"Linked to Navy summary line for {compound_name}.",
                )


def add_kcl_source(state: BuildState, source_id: str, path: Path, notes: str) -> None:
    add_source(state, source_id, path.name, "kcl_experimental_note_pdf", path, notes)


def import_kcl_notes(state: BuildState, kcl_dir: Path) -> None:
    melamine_update = kcl_dir / "Melamine 14N NQR - update.pdf"
    melamine_spectra = kcl_dir / "Melamine 14N NQR.pdf"
    metformin = kcl_dir / "Metformin HCL 14N NQR.pdf"
    paracetamol = kcl_dir / "Paracetamol 14N NQR.pdf"
    population = kcl_dir / "Population Transfer in a single-axis coil.pdf"

    add_kcl_source(state, "kcl_melamine_update_pdf", melamine_update, "Melamine 14N NQR line-frequency and quadrupolar-parameter update.")
    add_kcl_source(state, "kcl_melamine_spectra_pdf", melamine_spectra, "Melamine 14N NQR spectra note with PSL acquisition details.")
    add_kcl_source(state, "kcl_metformin_hcl_pdf", metformin, "Metformin HCl 14N NQR frequency and polarization-factor note.")
    add_kcl_source(state, "kcl_paracetamol_pdf", paracetamol, "Paracetamol 14N NQR spectra note.")
    add_kcl_source(state, "kcl_population_transfer_pdf", population, "Population-transfer method note; registered as provenance but not imported as spectral lines.")

    melamine_sites = [
        ("ring N", "2605", "0.96", [("nu+", "2582"), ("nu-", "1326"), ("nu0", "1256")]),
        ("ring N", "2588", "0.99", [("nu+", "2581"), ("nu-", "1301"), ("nu0", "1280")]),
        ("ring N", "2553", "0.93", [("nu+", "2507"), ("nu-", "1322"), ("nu0", "1185")]),
        ("amino N", "3313", "0.39", [("nu+", "2809"), ("nu-", "2160"), ("nu0", "649")]),
        ("amino N", "3200", "0.46", [("nu+", "2766"), ("nu-", "2034"), ("nu0", "732")]),
    ]
    for site_index, (label, qcc, eta, transitions) in enumerate(melamine_sites, start=1):
        original_site = {
            "curation_method": "manual_from_pdf_text",
            "source_pdf": melamine_update.name,
            "site_index": site_index,
        }
        site_id = upsert_site(
            state,
            "Melamine",
            "kcl_melamine_update_pdf",
            str(site_index),
            label,
            "14N",
            None,
            first_number(qcc),
            first_number(eta),
            original_site,
        )
        for transition_index, (transition, frequency) in enumerate(transitions, start=1):
            original = {**original_site, "transition": transition, "frequency": frequency}
            add_line_record(
                state,
                "kcl_melamine_update_pdf",
                site_id,
                frequency,
                original,
                row_index=site_index * 10 + transition_index,
                transition_label=transition,
            )

    metformin_rows = [
        ("3.068", "3.1", "NH"),
        ("2.908", "< 1", "N"),
        ("2.835", "7.9", "NH2"),
        ("2.821", "9.0", "NH2"),
        ("2.72", "3.8", "NH"),
        ("2.61", "19.8", "NH2"),
    ]
    for row_index, (freq_mhz, pol, group) in enumerate(metformin_rows, start=1):
        site_id = upsert_site(
            state,
            "Metformin HCl",
            "kcl_metformin_hcl_pdf",
            str(row_index),
            group,
            "14N",
            None,
            None,
            None,
            {"curation_method": "manual_from_pdf_table", "source_pdf": metformin.name, "row_index": row_index},
        )
        add_line_record(
            state,
            "kcl_metformin_hcl_pdf",
            site_id,
            str(float(freq_mhz) * 1000.0),
            {
                "curation_method": "manual_from_pdf_table",
                "source_pdf": metformin.name,
                "frequency_MHz": freq_mhz,
                "polarization_factor": pol,
                "predicted_functional_group": group,
            },
            row_index=row_index,
            polarization_factor_original=pol,
        )

    paracetamol_site = upsert_site(
        state,
        "Paracetamol",
        "kcl_paracetamol_pdf",
        "1",
        "14N",
        "14N",
        None,
        None,
        None,
        {"curation_method": "manual_from_pdf_text", "source_pdf": paracetamol.name},
    )
    for row_index, frequency in enumerate(["2564", "1921", "643"], start=1):
        add_line_record(
            state,
            "kcl_paracetamol_pdf",
            paracetamol_site,
            frequency,
            {
                "curation_method": "manual_from_pdf_text",
                "source_pdf": paracetamol.name,
                "note": "Lines reported as found by Ljubljana group.",
            },
            row_index=row_index,
        )
    paracetamol_detail_line_id = add_line_record(
        state,
        "kcl_paracetamol_pdf",
        paracetamol_site,
        "2563.7",
        {
            "curation_method": "manual_from_pdf_text",
            "source_pdf": paracetamol.name,
            "note": "Measured line at 297 K; T2* = 0.202 ms; temperature coefficient +70 Hz/K; T1 ca. 9 s.",
        },
        row_index=10,
        t1_original="ca 9 s",
        temperature_k=297.0,
        dnu_dt_original="+0.070 kHz/K",
    )
    if paracetamol_detail_line_id:
        state.lines[paracetamol_detail_line_id]["t2_star_s"] = 0.000202
        state.lines[paracetamol_detail_line_id]["t2_star_original"] = "0.202 ms"


def import_landolt_material(state: BuildState, landolt_dir: Path) -> None:
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pdfplumber is required for Landolt PDF extraction") from exc

    for source_id, filename in LANDOLT_PDFS.items():
        path = landolt_dir / filename
        add_source(
            state,
            source_id,
            filename,
            "landolt_bornstein_pdf",
            path,
            "Landolt-Bornstein NQR spectroscopy data excerpt.",
        )

    for definition_id, column_name, definition_text, source_page in LANDOLT_COLUMN_DEFINITIONS:
        row_id = "landolt_coldef:" + definition_id
        state.landolt_column_definitions[row_id] = {
            "id": row_id,
            "column_name": column_name,
            "definition_text": definition_text,
            "source_id": "landolt_intro_pdf",
            "source_page": source_page,
        }

    variables = {
        "e": "elementary charge",
        "q": "electric field gradient component",
        "Q": "nuclear quadrupole moment",
        "h": "Planck constant",
        "eta": "asymmetry parameter",
    }
    for row in LANDOLT_TRANSITION_EQUATIONS:
        state.nqr_transition_equations[row["id"]] = {
            "id": row["id"],
            "nuclear_spin": row["nuclear_spin"],
            "transition_label": row["transition_label"],
            "expression_text": row["expression_text"],
            "expression_latex": row["expression_latex"],
            "variables_json": json.dumps(variables, sort_keys=True),
            "source_id": "landolt_transition_frequencies_pdf",
            "source_page": "p. 16",
            "confidence": row["confidence"],
            "notes": row["notes"],
        }

    for source_id, filename in LANDOLT_PDFS.items():
        path = landolt_dir / filename
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or page.extract_text() or ""
                extract_id = "landolt_page:" + slug(f"{source_id}:{page_number}")
                state.landolt_page_extracts[extract_id] = {
                    "id": extract_id,
                    "source_id": source_id,
                    "source_page": page_number,
                    "extraction_method": "pdfplumber.extract_text(layout=True)",
                    "text": text,
                }

    for source_id, table_number in [
        ("landolt_nitrogen_table_a_pdf", "4"),
        ("landolt_nitrogen_table_b_pdf", "9"),
    ]:
        source_path = landolt_dir / LANDOLT_PDFS[source_id]
        import_landolt_nitrogen_entries(state, source_id, source_path, table_number)

    import_landolt_reference_codes(state, landolt_dir / LANDOLT_PDFS["landolt_nitrogen_references_pdf"])


def import_landolt_nitrogen_entries(
    state: BuildState,
    source_id: str,
    source_path: Path,
    table_number: str,
) -> None:
    import pdfplumber

    with pdfplumber.open(source_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or ""
            lines = text.splitlines()
            footnotes = extract_landolt_footnotes(lines)
            table_blocks = repair_landolt_table_blocks(source_id, page_number, extract_landolt_table_blocks(lines))
            for substance_number, raw_table_text in table_blocks.items():
                parsed = parse_landolt_table_block(raw_table_text)
                footnote = footnotes.get(substance_number, "")
                name, cas = parse_landolt_footnote(footnote)
                entry_id = "landolt_entry:" + slug(f"{source_id}:{substance_number}:{page_number}")
                state.landolt_compound_entries[entry_id] = {
                    "id": entry_id,
                    "source_id": source_id,
                    "source_page": page_number,
                    "table_number": table_number,
                    "substance_number": substance_number,
                    "formula_raw": parsed.get("formula_raw"),
                    "nucleus": parsed.get("nucleus"),
                    "method": parsed.get("method"),
                    "temperature_original": parsed.get("temperature_original"),
                    "frequencies_raw": parsed.get("frequencies_raw"),
                    "qcc_original": parsed.get("qcc_original"),
                    "eta_original": parsed.get("eta_original"),
                    "reference_code": parsed.get("reference_code"),
                    "remark_flag": parsed.get("remark_flag"),
                    "substance_name": name,
                    "cas_registry_number": cas,
                    "raw_table_text": raw_table_text,
                    "raw_footnote_text": footnote,
                    "extraction_confidence": parsed.get("extraction_confidence", "raw_ocr_layout"),
                    "notes": landolt_entry_notes(parsed),
                }


def extract_landolt_table_blocks(lines: list[str]) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current_number: str | None = None
    footnote_started = False
    for line in lines:
        stripped = line.rstrip()
        is_table_row = is_landolt_table_row_text(stripped)
        if re.match(LANDOLT_FOOTNOTE_NUMBER_PATTERN, stripped) and not is_table_row:
            footnote_started = True
        if footnote_started:
            continue
        match = re.match(LANDOLT_ROW_NUMBER_PATTERN, stripped)
        if match and is_table_row:
            current_number = match.group("num")
            blocks.setdefault(current_number, []).append(stripped)
        elif current_number and stripped.strip() and not re.search(r"^-{5,}", stripped.strip()):
            # Continuation lines usually hold additional frequencies/QCC/eta.
            if (
                re.match(r"^\s+\d", stripped)
                or re.match(r"^\s+[.-]?\d", stripped)
                or re.match(r"^\s+[CDPMEX]\s+", stripped)
            ):
                blocks[current_number].append(stripped)
    return {number: "\n".join(block) for number, block in blocks.items()}


def repair_landolt_table_blocks(source_id: str, page_number: int, blocks: dict[str, str]) -> dict[str, str]:
    if source_id == "landolt_nitrogen_table_b_pdf" and page_number == 4 and "118" in blocks:
        blocks = dict(blocks)
        blocks["118"] = "\n".join(
            [
                "118. C27H22N2P2PdS2 N-14 D 77.0 2.683 3.147 0.267 81FU1",
                "2.570 3.297 0.255",
                "0.42",
                "0.42",
            ]
        )
        blocks["119"] = "\n".join(
            [
                "119. C28H24N2P2PdS2 N-14 D 77.0 2.680 313.0 0.235 81FU1",
                "0.390 0.813 0.00",
            ]
        )
        blocks["120"] = "\n".join(
            [
                "120. C28H32Cl4N2O2P2Sn N-14 P 77.0 3.6157 4.7518 0.0436 75AN1",
                "3.5575 4.6002 0.0933",
                "3.5121",
                "3.3428",
            ]
        )
        blocks["121"] = "\n".join(
            [
                "121. C28H32N2(Achiral, Solid) N-14 D 293.0 3.439 4.220 0.260 * 79BL1",
                "2.891",
            ]
        )
    return blocks


def extract_landolt_footnotes(lines: list[str]) -> dict[str, str]:
    footnotes: dict[str, list[str]] = {}
    current_number: str | None = None
    for line in lines:
        stripped = normalize_space(line)
        match = re.match(LANDOLT_FOOTNOTE_NUMBER_PATTERN, stripped)
        if match:
            if is_landolt_table_row_text(stripped):
                current_number = None
                continue
            current_number = match.group("num")
            footnotes.setdefault(current_number, []).append(match.group("text"))
        elif current_number and stripped:
            if re.match(LANDOLT_ROW_NUMBER_PATTERN, stripped):
                current_number = None
            else:
                footnotes[current_number].append(stripped)
    return {number: " ".join(parts) for number, parts in footnotes.items()}


def parse_landolt_table_block(block: str) -> dict[str, str | None]:
    raw_block_lines = [line.rstrip() for line in block.splitlines() if normalize_space(line)]
    block_lines = [normalize_space(line) for line in raw_block_lines]
    first_line = block_lines[0] if block_lines else ""
    number_match = re.match(LANDOLT_ROW_NUMBER_PATTERN, first_line)
    if not number_match:
        return {"extraction_confidence": "raw_ocr_layout"}
    after_number = number_match.group("after_number")
    nucleus_match = re.search(r"\b(?P<nucleus>N-14|14)\b", after_number)
    if not nucleus_match:
        nucleus_match = re.search(r"\b(?P<nucleus>[A-Z][a-z]?-?\d+|14)\b", after_number)
    if not nucleus_match:
        return {"extraction_confidence": "raw_ocr_layout"}
    formula_raw = normalize_landolt_formula(after_number[: nucleus_match.start()].strip()) or None
    rest = after_number[nucleus_match.end() :].strip()
    tokens = " ".join([rest, *block_lines[1:]]).split()
    method = None
    temperature = None
    first_rest_tokens = rest.split()
    if first_rest_tokens and first_rest_tokens[0] in LANDOLT_METHOD_DEFINITIONS:
        method = first_rest_tokens.pop(0)
    if first_rest_tokens and (
        re.fullmatch(r"\d+(?:\.\d+)?", first_rest_tokens[0])
        or is_landolt_room_temperature_token(first_rest_tokens[0])
    ):
        temperature = first_rest_tokens.pop(0)
    reference_code = None
    reference_code = find_landolt_reference_code(tokens)
    remark_flag = "*" if "*" in tokens else None
    frequencies: list[str] = []
    qcc_values: list[str] = []
    eta_values: list[str] = []
    numeric_rows = [first_rest_tokens, *[landolt_legacy_numeric_row_tokens(line) for line in raw_block_lines[1:]]]
    for row_tokens in numeric_rows:
        numeric_tokens = [token for token in row_tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token)]
        if numeric_tokens:
            frequencies.append(numeric_tokens[0])
        if len(numeric_tokens) >= 2:
            qcc_values.append(numeric_tokens[1])
        if len(numeric_tokens) >= 3:
            eta_values.append(numeric_tokens[2])
    return {
        "formula_raw": formula_raw,
        "nucleus": "N-14" if nucleus_match.group("nucleus") == "14" else nucleus_match.group("nucleus"),
        "method": method,
        "temperature_original": temperature,
        "frequencies_raw": "; ".join(frequencies) or None,
        "qcc_original": "; ".join(qcc_values) or None,
        "eta_original": "; ".join(eta_values) or None,
        "reference_code": reference_code,
        "remark_flag": remark_flag,
        "extraction_confidence": "parsed_layout_columns_raw_values",
    }


def landolt_legacy_numeric_row_tokens(line: str) -> list[str]:
    tokens = normalize_space(line).split()
    if tokens and tokens[0] in LANDOLT_METHOD_DEFINITIONS:
        tokens.pop(0)
        if tokens and (
            re.fullmatch(r"\d+(?:\.\d+)?", tokens[0])
            or is_landolt_room_temperature_token(tokens[0])
        ):
            tokens.pop(0)
    elif landolt_starts_in_temperature_column(line) and len(landolt_numeric_tokens(tokens)) >= 4:
        tokens.pop(0)
    return tokens


def landolt_numeric_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token)]


def landolt_starts_in_temperature_column(line: str) -> bool:
    stripped = line.strip()
    if not re.match(r"[-+]?\d+(?:\.\d+)?(?:\s|$)", stripped):
        return False
    leading_spaces = len(line) - len(line.lstrip(" "))
    return leading_spaces <= 56


def landolt_entry_notes(parsed: dict[str, str | None]) -> str:
    notes = ["Parsed from Landolt layout text; verify against page image before promoting to canonical line records."]
    temperature = parsed.get("temperature_original")
    if temperature and is_landolt_room_temperature_token(temperature):
        notes.append(f"Temperature token {temperature!r} means room temperature; exact numeric temperature is not specified by this token.")
    return " ".join(notes)


def is_landolt_room_temperature_token(value: str) -> bool:
    return normalize_space(value).replace(".", "").lower() in {"rt", "rtemp"}


def is_landolt_table_row_text(text: str) -> bool:
    return bool(
        re.match(
            r"^[\s'`\u2018\u2019]*\d{2,3}\.?\s+(?:(?!\b(?:N-14|14)\b)\S+\s+){0,3}(?:N-14|14)\b",
            text,
        )
    )


def parse_landolt_footnote(footnote: str) -> tuple[str | None, str | None]:
    if not footnote:
        return None, None
    cas_match = re.search(r"[CEtT]\d{2,7}-\d{2}-\d{1,2}", footnote)
    cas = None
    if cas_match:
        cas = cas_match.group(0)[1:] if cas_match.group(0)[0] in {"C", "E", "t", "T"} else cas_match.group(0)
        cas = normalize_landolt_cas(cas)
    name = footnote
    if cas_match:
        name = footnote[: cas_match.start()].strip(" -")
    name = re.sub(r"\s+\*.*$", "", name).strip()
    return name or None, cas


def normalize_landolt_cas(value: str) -> str:
    match = re.fullmatch(r"(\d{2,7}-\d{2}-\d)(?:1)?", value.strip())
    return match.group(1) if match else value.strip()


def normalize_landolt_formula(value: str | None) -> str | None:
    if not value:
        return None
    text = normalize_landolt_formula_tokens(value.replace(" ", "").replace("Q", "O"))
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"C1(?=[A-Z]|$)", "Cl", text)
        text = normalize_landolt_formula_tokens(text)
    return text


def normalize_landolt_formula_tokens(value: str) -> str:
    parts: list[str] = []
    pos = 0
    for match in re.finditer(r"[A-Z][a-z]?\d*", value):
        parts.append(value[pos : match.start()])
        token = match.group(0)
        element_match = re.match(r"([A-Z][a-z]?)(\d*)", token)
        if not element_match:
            parts.append(token)
        else:
            element, count = element_match.groups()
            if "0" in count and not count.endswith("0"):
                zero_index = count.find("0")
                before = count[:zero_index]
                after = count[zero_index + 1 :]
                parts.append(f"{element}{before}O{after}")
            elif count.startswith("0"):
                parts.append(f"{element}O{count[1:]}")
            else:
                parts.append(token)
        pos = match.end()
    parts.append(value[pos:])
    return "".join(parts)


def import_landolt_reference_codes(state: BuildState, reference_pdf: Path) -> None:
    import pdfplumber

    with pdfplumber.open(reference_pdf) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
            pairs = extract_landolt_reference_code_pairs_from_words(words, page.width)
            if not pairs:
                text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=3) or page.extract_text() or ""
                pairs = extract_landolt_reference_code_pairs(text)
            for code, citation in pairs:
                row_id = "landolt_refcode:" + slug(f"{code}:{citation[:80]}")
                state.landolt_reference_codes[row_id] = {
                    "id": row_id,
                    "source_id": "landolt_nitrogen_references_pdf",
                    "table_number": "9",
                    "reference_code": code,
                    "citation_text": citation,
                    "source_page": f"p. {page_number}",
                    "extraction_confidence": "word_coordinates_two_column",
                }


def extract_landolt_reference_code_pairs_from_words(
    words: list[dict[str, Any]], page_width: float
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    split_x = page_width / 2
    for column_min, column_max in [(0.0, split_x), (split_x, page_width + 1.0)]:
        column_words = [
            word
            for word in words
            if word.get("top", 0) > 35
            and column_min <= (float(word["x0"]) + float(word["x1"])) / 2 < column_max
        ]
        current_code: str | None = None
        current_parts: list[str] = []
        for line_words in group_words_into_lines(column_words):
            if not line_words:
                continue
            code, rest = pop_landolt_reference_code(line_words)
            if code:
                if current_code and current_parts:
                    citation = normalize_space(" ".join(current_parts))
                    if len(citation) > 10:
                        pairs.append((current_code, citation))
                current_code = code
                current_parts = [rest] if rest else []
            elif current_code:
                current_parts.append(" ".join(str(word["text"]) for word in line_words))
        if current_code and current_parts:
            citation = normalize_space(" ".join(current_parts))
            if len(citation) > 10:
                pairs.append((current_code, citation))
    return pairs


def group_words_into_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    line_tops: list[float] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not lines or abs(top - line_tops[-1]) > 3.5:
            lines.append([word])
            line_tops.append(top)
        else:
            lines[-1].append(word)
            line_tops[-1] = (line_tops[-1] + top) / 2
    return [sorted(line, key=lambda item: float(item["x0"])) for line in lines]


def pop_landolt_reference_code(line_words: list[dict[str, Any]]) -> tuple[str | None, str]:
    tokens = [str(word["text"]).strip() for word in line_words if str(word["text"]).strip()]
    if not tokens:
        return None, ""
    candidate = tokens[0].strip(".,;:")
    rest_start = 1
    if (
        len(candidate) == 4
        and len(tokens) > 1
        and re.fullmatch(r"[0-9OoIlL]", tokens[1].strip(".,;:"))
    ):
        candidate += tokens[1].strip(".,;:")
        rest_start = 2
    code = normalize_landolt_reference_code(candidate)
    if not code:
        return None, ""
    return code, " ".join(tokens[rest_start:]).strip()


def find_landolt_reference_code(tokens: list[str]) -> str | None:
    for idx in range(len(tokens) - 1, -1, -1):
        code = normalize_landolt_reference_code_token(tokens[idx])
        if code:
            return code
        if idx + 1 < len(tokens):
            code = normalize_landolt_reference_code_token(tokens[idx] + tokens[idx + 1])
            if code:
                return code
    return None


def normalize_landolt_reference_code_token(raw_code: str) -> str | None:
    if "." in raw_code or "," in raw_code:
        return None
    raw = re.sub(r"[^0-9A-Za-z]", "", raw_code)
    if not re.search(r"[A-Za-z]", raw):
        return None
    return normalize_landolt_reference_code(raw)


def normalize_landolt_reference_code(raw_code: str) -> str | None:
    raw = re.sub(r"[^0-9A-Za-z]", "", raw_code).upper()
    if len(raw) != 5:
        return None
    digit_map = {"O": "0", "Q": "0", "I": "1", "L": "1"}
    letter_map = {"0": "O", "5": "S", "1": "I"}
    chars = list(raw)
    for idx in [0, 1, 4]:
        chars[idx] = digit_map.get(chars[idx], chars[idx])
        if not chars[idx].isdigit():
            return None
    for idx in [2, 3]:
        chars[idx] = letter_map.get(chars[idx], chars[idx])
        if not chars[idx].isalpha():
            return None
    return "".join(chars)


def extract_landolt_reference_code_pairs(text: str) -> list[tuple[str, str]]:
    normalized = re.sub(r"\s+", " ", text)
    pattern = re.compile(r"(\d{2}[A-Z]{2}\d)\s+")
    matches = list(pattern.finditer(normalized))
    pairs: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        citation = normalized[start:end].strip(" ;")
        if citation and len(citation) > 10:
            pairs.append((match.group(1), citation))
    return pairs


def build_landolt_independent_value_records(state: BuildState) -> None:
    state.landolt_measurement_sets.clear()
    state.landolt_frequency_records.clear()
    state.landolt_qcc_eta_records.clear()
    for entry in state.landolt_compound_entries.values():
        measurement_sets = parse_landolt_measurement_sets(entry)
        for set_index, measurement_set in enumerate(measurement_sets, start=1):
            set_id = "landolt_measurement_set:" + slug(f"{entry['id']}:{set_index}")
            state.landolt_measurement_sets[set_id] = {
                "id": set_id,
                "entry_id": entry["id"],
                "source_id": entry["source_id"],
                "source_page": entry["source_page"],
                "table_number": entry["table_number"],
                "substance_number": entry["substance_number"],
                "set_index": set_index,
                "method": measurement_set["method"],
                "method_description": LANDOLT_METHOD_DEFINITIONS.get(measurement_set["method"] or ""),
                "temperature_original": measurement_set["temperature_original"],
                "reference_code": measurement_set["reference_code"],
                "remark_flag": measurement_set["remark_flag"],
                "raw_set_text": "\n".join(measurement_set["raw_lines"]),
                "notes": landolt_measurement_set_notes(measurement_set),
            }
            for frequency_index, frequency in enumerate(measurement_set["frequencies"], start=1):
                row_id = "landolt_freq:" + slug(f"{set_id}:{frequency_index}:{frequency}")
                state.landolt_frequency_records[row_id] = {
                    "id": row_id,
                    "entry_id": entry["id"],
                    "measurement_set_id": set_id,
                    "source_id": entry["source_id"],
                    "source_page": entry["source_page"],
                    "table_number": entry["table_number"],
                    "substance_number": entry["substance_number"],
                    "sequence_index": frequency_index,
                    "frequency_original": frequency,
                    "notes": "Independent Landolt frequency record within this measurement set; no assignment to Q.C.C./eta records is inferred.",
                }
            for qcc_eta_index, pair in enumerate(measurement_set["qcc_eta_pairs"], start=1):
                row_id = "landolt_qcc_eta:" + slug(
                    f"{set_id}:{qcc_eta_index}:{pair.get('qcc_original') or ''}:{pair.get('eta_original') or ''}"
                )
                state.landolt_qcc_eta_records[row_id] = {
                    "id": row_id,
                    "entry_id": entry["id"],
                    "measurement_set_id": set_id,
                    "source_id": entry["source_id"],
                    "source_page": entry["source_page"],
                    "table_number": entry["table_number"],
                    "substance_number": entry["substance_number"],
                    "sequence_index": qcc_eta_index,
                    "qcc_original": pair.get("qcc_original"),
                    "eta_original": pair.get("eta_original"),
                    "notes": "Independent Landolt Q.C.C./eta record within this measurement set; no assignment to frequency records is inferred.",
                }


def parse_landolt_measurement_sets(entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw_lines = [line.rstrip() for line in (entry.get("raw_table_text") or "").splitlines() if normalize_space(line)]
    measurement_sets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_index, line in enumerate(raw_lines):
        row_tokens = landolt_measurement_row_tokens(line, line_index == 0)
        starts_new_set = bool(row_tokens and row_tokens[0] in LANDOLT_METHOD_DEFINITIONS)
        if current and landolt_starts_in_temperature_column(line) and len(landolt_numeric_tokens(row_tokens)) >= 4:
            starts_new_set = True
        if current is None or starts_new_set:
            if current:
                measurement_sets.append(current)
            current = {
                "method": None,
                "temperature_original": None,
                "reference_code": None,
                "remark_flag": None,
                "raw_lines": [],
                "frequencies": [],
                "qcc_eta_pairs": [],
            }
        if current is None:
            continue
        current["raw_lines"].append(normalize_space(line))
        parse_landolt_measurement_row_into_set(current, row_tokens)
    if current:
        measurement_sets.append(current)
    if measurement_sets:
        return measurement_sets

    fallback = {
        "method": entry.get("method"),
        "temperature_original": entry.get("temperature_original"),
        "reference_code": entry.get("reference_code"),
        "remark_flag": entry.get("remark_flag"),
        "raw_lines": raw_lines,
        "frequencies": split_landolt_value_list(entry.get("frequencies_raw")),
        "qcc_eta_pairs": [
            {"qcc_original": qcc, "eta_original": eta}
            for qcc, eta in zip_longest_lists(
                split_landolt_value_list(entry.get("qcc_original")),
                split_landolt_value_list(entry.get("eta_original")),
            )
        ],
    }
    return [fallback]


def landolt_measurement_row_tokens(line: str, is_first_line: bool) -> list[str]:
    if is_first_line:
        number_match = re.match(LANDOLT_ROW_NUMBER_PATTERN, normalize_space(line))
        after_number = number_match.group("after_number") if number_match else line
        nucleus_match = re.search(r"\b(?:N-14|14)\b", after_number)
        if not nucleus_match:
            nucleus_match = re.search(r"\b(?:[A-Z][a-z]?-?\d+|14)\b", after_number)
        return after_number[nucleus_match.end() :].strip().split() if nucleus_match else after_number.split()
    return normalize_space(line).split()


def parse_landolt_measurement_row_into_set(measurement_set: dict[str, Any], tokens: list[str]) -> None:
    tokens = list(tokens)
    if not tokens:
        return
    saw_method = False
    if tokens and tokens[0] in LANDOLT_METHOD_DEFINITIONS:
        measurement_set["method"] = tokens.pop(0)
        saw_method = True
    expects_temperature = saw_method or (
        not measurement_set.get("temperature_original")
        and not measurement_set.get("frequencies")
        and not measurement_set.get("qcc_eta_pairs")
    )
    if expects_temperature and tokens and (
        re.fullmatch(r"\d+(?:\.\d+)?", tokens[0])
        or is_landolt_room_temperature_token(tokens[0])
    ):
        measurement_set["temperature_original"] = tokens.pop(0)
    reference_code = find_landolt_reference_code(tokens)
    if reference_code:
        measurement_set["reference_code"] = reference_code
    if "*" in tokens:
        measurement_set["remark_flag"] = "*"
    numeric_tokens = [token for token in tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token)]
    if numeric_tokens:
        measurement_set["frequencies"].append(numeric_tokens[0])
    if len(numeric_tokens) >= 2:
        measurement_set["qcc_eta_pairs"].append(
            {
                "qcc_original": numeric_tokens[1],
                "eta_original": numeric_tokens[2] if len(numeric_tokens) >= 3 else None,
            }
        )


def zip_longest_lists(left: list[str], right: list[str]) -> list[tuple[str | None, str | None]]:
    return [
        (
            left[index] if index < len(left) else None,
            right[index] if index < len(right) else None,
        )
        for index in range(max(len(left), len(right)))
    ]


def landolt_measurement_set_notes(measurement_set: dict[str, Any]) -> str:
    notes = ["Landolt measurement set; frequencies and Q.C.C./eta pairs are independent lists within this set."]
    method = measurement_set.get("method")
    if method and method in LANDOLT_METHOD_DEFINITIONS:
        notes.append(f"Method {method}: {LANDOLT_METHOD_DEFINITIONS[method]}.")
    temperature = measurement_set.get("temperature_original")
    if temperature and is_landolt_room_temperature_token(temperature):
        notes.append(f"Temperature token {temperature!r} means room temperature; exact numeric temperature is not specified by this token.")
    return " ".join(notes)


def split_landolt_value_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r"[;\n,]+", value) if part.strip()]


def promote_accepted_landolt_reviews(state: BuildState, decisions_path: Path = LANDOLT_REVIEW_DECISIONS) -> None:
    decisions = latest_landolt_review_decisions(decisions_path)
    if not decisions:
        return
    review_rows = {row["id"]: row for row in state.landolt_review_queue.values()}
    reference_codes = landolt_reference_code_lookup(state)
    for review_id, decision in sorted(decisions.items()):
        if decision.get("status") != "accepted":
            continue
        review_row = review_rows.get(review_id)
        if not review_row:
            continue
        entry = state.landolt_compound_entries.get(review_row["entry_id"])
        if not entry:
            continue
        promoted_entry = reviewed_landolt_entry(entry, decision)
        measurement_sets = decision.get("measurement_sets") or []
        if not measurement_sets:
            continue
        promote_landolt_entry(state, promoted_entry, decision, measurement_sets, reference_codes)


def latest_landolt_review_decisions(decisions_path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    if not decisions_path.exists():
        return decisions
    with decisions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            decisions[record["review_id"]] = record
    return decisions


def reviewed_landolt_entry(entry: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    promoted = dict(entry)
    for field, value in (decision.get("field_edits") or {}).items():
        promoted[field] = value
    return promoted


def landolt_reference_code_lookup(state: BuildState) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in state.landolt_reference_codes.values():
        code = normalize_landolt_reference_code(str(row["reference_code"]))
        if code and code not in lookup:
            lookup[code] = row
    return lookup


def promote_landolt_entry(
    state: BuildState,
    entry: dict[str, Any],
    decision: dict[str, Any],
    measurement_sets: list[dict[str, Any]],
    reference_codes: dict[str, dict[str, Any]],
) -> None:
    name = normalize_space(entry.get("substance_name")) or normalize_space(entry.get("formula_raw")) or (
        f"Landolt Table {entry.get('table_number')} Substance {entry.get('substance_number')}"
    )
    formula = normalize_space(entry.get("formula_raw")) or None
    compound_id = ensure_landolt_compound(state, name, formula, entry)
    if formula:
        state.aliases.add((compound_id, formula))
    cas = normalize_space(entry.get("cas_registry_number"))
    if cas:
        state.aliases.add((compound_id, f"CAS {cas}"))
    state.aliases.add((compound_id, f"Landolt Table {entry.get('table_number')} Substance {entry.get('substance_number')}"))

    previous_reference_code = normalize_space(entry.get("reference_code")) or None
    for set_index, measurement_set in enumerate(measurement_sets, start=1):
        set_reference_code = normalize_space(measurement_set.get("reference_code")) or previous_reference_code
        if set_reference_code:
            previous_reference_code = set_reference_code
        sample_id = add_landolt_sample(state, compound_id, entry, measurement_set, set_index)
        frequency_site_id = add_landolt_frequency_site(state, sample_id, entry, measurement_set, set_index)
        promoted_line_ids = promote_landolt_frequency_records(
            state,
            frequency_site_id,
            entry,
            decision,
            measurement_set,
            set_index,
        )
        qcc_site_ids = promote_landolt_qcc_eta_records(
            state,
            sample_id,
            entry,
            decision,
            measurement_set,
            set_index,
        )
        reference_id = promote_landolt_reference_code(
            state,
            set_reference_code,
            reference_codes,
            entry,
            name,
        )
        if reference_id:
            add_reference_link(
                state,
                reference_id,
                str(entry["source_id"]),
                "compound",
                compound_id=compound_id,
                note="Landolt reviewed compound entry.",
            )
            for line_id in promoted_line_ids:
                add_reference_link(
                    state,
                    reference_id,
                    str(entry["source_id"]),
                    "line",
                    line_id=line_id,
                    note="Landolt measurement-set reference code.",
                )
            for site_id in qcc_site_ids:
                add_reference_link(
                    state,
                    reference_id,
                    str(entry["source_id"]),
                    "site",
                    site_id=site_id,
                    note="Landolt Q.C.C./eta reference code.",
                )


def ensure_landolt_compound(
    state: BuildState, name: str, formula: str | None, entry: dict[str, Any]
) -> str:
    compound_id = ensure_compound(state, name, "landolt")
    compound = state.compounds[compound_id]
    if formula:
        compound["formula"] = compound.get("formula") or formula
        compound["conventional_formula"] = compound.get("conventional_formula") or formula
    compound["category"] = compound.get("category") or "landolt"
    note = f"Includes reviewed Landolt Table {entry.get('table_number')} substance {entry.get('substance_number')} data."
    if compound.get("notes"):
        if note not in compound["notes"]:
            compound["notes"] += " " + note
    else:
        compound["notes"] = note
    return compound_id


def add_landolt_sample(
    state: BuildState,
    compound_id: str,
    entry: dict[str, Any],
    measurement_set: dict[str, Any],
    set_index: int,
) -> str:
    sample_id = "sample:" + slug(f"landolt:{entry['id']}:{set_index}")
    method = normalize_space(measurement_set.get("method")) or normalize_space(entry.get("method")) or None
    method_description = normalize_space(measurement_set.get("method_description")) or (
        LANDOLT_METHOD_DEFINITIONS.get(method or "") if method else None
    )
    temperature_original = normalize_space(measurement_set.get("temperature_original")) or None
    state.samples[sample_id] = {
        "id": sample_id,
        "compound_id": compound_id,
        "label": f"Landolt Table {entry.get('table_number')} substance {entry.get('substance_number')} set {set_index}",
        "form": None,
        "phase": None,
        "temperature_k": landolt_temperature_k(temperature_original),
        "notes": normalize_space(
            " ".join(
                part
                for part in [
                    "Reviewed Landolt measurement set.",
                    f"Method {method}: {method_description}." if method and method_description else None,
                    f"Temperature original: {temperature_original}." if temperature_original else None,
                ]
                if part
            )
        )
        or None,
    }
    return sample_id


def add_landolt_frequency_site(
    state: BuildState,
    sample_id: str,
    entry: dict[str, Any],
    measurement_set: dict[str, Any],
    set_index: int,
) -> str:
    original = {
        "landolt_entry_id": entry["id"],
        "table_number": entry.get("table_number"),
        "substance_number": entry.get("substance_number"),
        "measurement_set": measurement_set_without_children(measurement_set),
        "promotion_note": "Frequencies are not assigned to Q.C.C./eta pairs in the Landolt review workflow.",
    }
    site_id = "site:" + slug(f"landolt:{entry['id']}:{set_index}:frequency-list")
    state.sites[site_id] = {
        "id": site_id,
        "sample_id": sample_id,
        "site_number": None,
        "isotope": isotope_from_landolt_nucleus(entry.get("nucleus")),
        "site_label": f"Landolt set {set_index} frequency list",
        "weight_percent": None,
        "qcc_khz": None,
        "eta": None,
        "assignment_confidence": "unassigned_frequency_list",
        "source_id": entry["source_id"],
        "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }
    return site_id


def promote_landolt_frequency_records(
    state: BuildState,
    site_id: str,
    entry: dict[str, Any],
    decision: dict[str, Any],
    measurement_set: dict[str, Any],
    set_index: int,
) -> list[str]:
    line_ids: list[str] = []
    temperature_original = normalize_space(measurement_set.get("temperature_original")) or None
    for sequence_index, frequency in enumerate(measurement_set.get("frequency_records") or [], start=1):
        frequency_original = normalize_space(frequency.get("frequency_original"))
        frequency_khz = landolt_mhz_to_khz(frequency_original)
        if frequency_khz is None:
            continue
        original = {
            "landolt_entry_id": entry["id"],
            "review_id": decision.get("review_id"),
            "table_number": entry.get("table_number"),
            "substance_number": entry.get("substance_number"),
            "measurement_set": measurement_set_without_children(measurement_set),
            "frequency_record": frequency,
            "promotion_note": "Landolt frequencies are stored as an independent list within the measurement set.",
        }
        line_id = "line:" + slug(f"landolt:{entry['id']}:{set_index}:freq:{sequence_index}:{frequency_original}")
        state.lines[line_id] = {
            "id": line_id,
            "site_id": site_id,
            "frequency_khz": frequency_khz,
            "frequency_original": frequency_original,
            "transition_label": None,
            "fwhm_khz": None,
            "line_width_khz": None,
            "line_width_original": None,
            "t1_s": None,
            "t1_original": None,
            "t2_s": None,
            "t2_original": None,
            "t2_star_s": None,
            "t2_star_original": None,
            "dnu_dt_khz_per_c": None,
            "dnu_dt_original": None,
            "polarization_factor": None,
            "polarization_factor_original": None,
            "temperature_k": landolt_temperature_k(temperature_original),
            "form": None,
            "source_id": entry["source_id"],
            "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
        }
        line_ids.append(line_id)
    return line_ids


def promote_landolt_qcc_eta_records(
    state: BuildState,
    sample_id: str,
    entry: dict[str, Any],
    decision: dict[str, Any],
    measurement_set: dict[str, Any],
    set_index: int,
) -> list[str]:
    site_ids: list[str] = []
    for sequence_index, pair in enumerate(measurement_set.get("qcc_eta_records") or [], start=1):
        qcc_original = normalize_space(pair.get("qcc_original"))
        eta_original = normalize_space(pair.get("eta_original"))
        qcc_khz = landolt_mhz_to_khz(qcc_original)
        eta = first_number(eta_original)
        original = {
            "landolt_entry_id": entry["id"],
            "review_id": decision.get("review_id"),
            "table_number": entry.get("table_number"),
            "substance_number": entry.get("substance_number"),
            "measurement_set": measurement_set_without_children(measurement_set),
            "qcc_eta_record": pair,
            "promotion_note": "Q.C.C./eta pairs are not assigned to frequency records in the Landolt review workflow.",
        }
        site_id = "site:" + slug(f"landolt:{entry['id']}:{set_index}:qcc-eta:{sequence_index}:{qcc_original}:{eta_original}")
        state.sites[site_id] = {
            "id": site_id,
            "sample_id": sample_id,
            "site_number": str(sequence_index),
            "isotope": isotope_from_landolt_nucleus(entry.get("nucleus")),
            "site_label": f"Landolt set {set_index} Q.C.C./eta pair {sequence_index}",
            "weight_percent": None,
            "qcc_khz": qcc_khz,
            "eta": eta,
            "assignment_confidence": "source_reported_unassigned_to_lines",
            "source_id": entry["source_id"],
            "original_record": json.dumps(original, ensure_ascii=False, sort_keys=True),
        }
        site_ids.append(site_id)
    return site_ids


def promote_landolt_reference_code(
    state: BuildState,
    reference_code: str | None,
    reference_codes: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    compound_name: str,
) -> str | None:
    code = normalize_landolt_reference_code(reference_code or "")
    if not code:
        return None
    row = reference_codes.get(code)
    if row:
        citation_text = row["citation_text"]
        source_page = row["source_page"]
    else:
        citation_text = f"Unresolved Landolt reference code {code}"
        source_page = None
    reference_id = "ref:" + slug(f"landolt:{code}:{citation_text[:80]}")
    state.literature_references.setdefault(
        reference_id,
        {
            "id": reference_id,
            "citation_text": citation_text,
            "reference_type": "landolt_reference_code",
            "authors": None,
            "year": extract_year(citation_text),
            "title": None,
            "journal": None,
            "doi": None,
            "source_id": "landolt_nitrogen_references_pdf",
            "source_page": source_page or f"Landolt Table {entry.get('table_number')}",
            "source_reference_number": code,
            "original_text": citation_text,
        },
    )
    return reference_id


def measurement_set_without_children(measurement_set: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in measurement_set.items()
        if key not in {"frequency_records", "qcc_eta_records"}
    }


def isotope_from_landolt_nucleus(value: str | None) -> str | None:
    text = normalize_space(value)
    if text == "N-14":
        return "14N"
    match = re.fullmatch(r"([A-Z][a-z]?)-?(\d{1,3})", text)
    return f"{match.group(2)}{match.group(1)}" if match else text or None


def landolt_temperature_k(value: str | None) -> float | None:
    text = normalize_space(value)
    if not text or is_landolt_room_temperature_token(text):
        return None
    return first_number(text)


def landolt_mhz_to_khz(value: str | None) -> float | None:
    number = first_number(value)
    return number * 1000.0 if number is not None else None


def build_landolt_review_queue(
    state: BuildState,
    landolt_dir: Path,
    crop_dir: Path = LANDOLT_CROP_DIR,
    generate_crops: bool = True,
) -> None:
    import pdfplumber

    crop_dir.mkdir(parents=True, exist_ok=True)
    if generate_crops:
        clear_landolt_crop_dir(crop_dir)
    decisions = latest_landolt_review_decisions(LANDOLT_REVIEW_DECISIONS)
    reference_codes = {row["reference_code"] for row in state.landolt_reference_codes.values()}
    entries_by_source_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in state.landolt_compound_entries.values():
        source_page = entry.get("source_page")
        if source_page is None:
            continue
        entries_by_source_page.setdefault((entry["source_id"], int(source_page)), []).append(entry)

    pdfium = None
    if generate_crops:
        try:
            import pypdfium2 as pdfium_module
        except Exception as exc:  # pragma: no cover - optional rendering guard
            raise RuntimeError("pypdfium2 is required to generate Landolt review crops") from exc
        pdfium = pdfium_module

    for source_id in ["landolt_nitrogen_table_a_pdf", "landolt_nitrogen_table_b_pdf"]:
        source_path = landolt_dir / LANDOLT_PDFS[source_id]
        pdfium_doc = pdfium.PdfDocument(str(source_path)) if pdfium else None
        try:
            with pdfplumber.open(source_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_entries = entries_by_source_page.get((source_id, page_number), [])
                    if not page_entries:
                        continue
                    regions = extract_landolt_review_regions(page)
                    rendered_page = None
                    for entry in page_entries:
                        region = regions.get(str(entry["substance_number"]))
                        crop_relative_path = None
                        crop_bbox_json = None
                        if generate_crops and region and pdfium_doc:
                            if rendered_page is None:
                                rendered_page = render_pdfium_page(pdfium_doc, page_number, scale=3.0)
                            crop_relative_path = write_landolt_review_crop(
                                rendered_page,
                                page.width,
                                region,
                                entry,
                                crop_dir,
                                scale=3.0,
                            )
                            crop_bbox_json = json.dumps(region, sort_keys=True)
                        issue_flags = landolt_review_issue_flags(entry, region, reference_codes)
                        review_id = "landolt_review:" + slug(entry["id"])
                        decision = decisions.get(review_id) or {}
                        state.landolt_review_queue[review_id] = {
                            "id": review_id,
                            "entry_id": entry["id"],
                            "status": decision.get("status") or "unreviewed",
                            "priority": landolt_review_priority(issue_flags),
                            "issue_flags_json": json.dumps(issue_flags, sort_keys=True),
                            "crop_relative_path": crop_relative_path,
                            "crop_bbox_json": crop_bbox_json,
                            "source_id": source_id,
                            "source_page": page_number,
                            "reviewer_notes": decision.get("reviewer_notes"),
                            "promoted_line_id": None,
                            "updated_at": decision.get("updated_at"),
                        }
        finally:
            if pdfium_doc is not None:
                pdfium_doc.close()


def extract_landolt_review_regions(page: Any) -> dict[str, dict[str, Any]]:
    words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
    line_words = group_words_into_lines(words)
    line_records = [landolt_line_record(line) for line in line_words if line]
    table_regions = repair_landolt_table_regions(extract_landolt_table_regions(line_records), line_records)
    footnote_regions = extract_landolt_footnote_regions(line_records)
    regions: dict[str, dict[str, Any]] = {}
    for substance_number, table_region in table_regions.items():
        footnote_region = footnote_regions.get(substance_number)
        regions[substance_number] = {
            "table_bbox": table_region["bbox"],
            "footnote_bbox": footnote_region["bbox"] if footnote_region else None,
            "table_text": table_region["text"],
            "footnote_text": footnote_region["text"] if footnote_region else None,
        }
    return regions


def landolt_line_record(line_words: list[dict[str, Any]]) -> dict[str, Any]:
    text = normalize_space(" ".join(str(word["text"]) for word in line_words))
    return {
        "text": text,
        "bbox": [
            min(float(word["x0"]) for word in line_words),
            min(float(word["top"]) for word in line_words),
            max(float(word["x1"]) for word in line_words),
            max(float(word["bottom"]) for word in line_words),
        ],
    }


def extract_landolt_table_regions(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    current_number: str | None = None
    footnote_started = False
    for line in lines:
        stripped = line["text"]
        is_table_row = is_landolt_table_row_text(stripped)
        if re.match(LANDOLT_FOOTNOTE_NUMBER_PATTERN, stripped) and not is_table_row:
            footnote_started = True
        if footnote_started:
            continue
        match = re.match(LANDOLT_ROW_NUMBER_PATTERN, stripped)
        if match and is_table_row:
            current_number = match.group("num")
            regions[current_number] = {"text": stripped, "bbox": line["bbox"]}
        elif current_number and stripped and not re.search(r"^-{5,}", stripped):
            if re.match(r"^[.-]?\d", stripped):
                regions[current_number]["text"] += "\n" + stripped
                regions[current_number]["bbox"] = union_bbox(regions[current_number]["bbox"], line["bbox"])
    return regions


def repair_landolt_table_regions(
    regions: dict[str, dict[str, Any]], lines: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    line_texts = [line["text"] for line in lines]
    if not any("C28H24N2P2PdS2" in text for text in line_texts):
        return regions
    repaired = dict(regions)
    specs = {
        "118": ("118.", ["2.570", "0.42"]),
        "119": ("C28H24N2P2PdS2", ["0.390"]),
        "120": ("3.6157", ["3.5575", "3.5121", "3.3428"]),
        "121": ("C28H32N2(Achiral", ["2.891"]),
    }
    for number, (anchor, continuations) in specs.items():
        anchor_index = next((idx for idx, line in enumerate(lines) if anchor in line["text"]), None)
        if anchor_index is None:
            continue
        region = {"text": lines[anchor_index]["text"], "bbox": lines[anchor_index]["bbox"]}
        for line in lines[anchor_index + 1 :]:
            text = line["text"]
            if any(stop in text for stop in ("119.", "120.", "121.", "---", "116.")):
                break
            if any(marker in text for marker in continuations):
                region["text"] += "\n" + text
                region["bbox"] = union_bbox(region["bbox"], line["bbox"])
        repaired[number] = region
    return repaired


def extract_landolt_footnote_regions(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    current_number: str | None = None
    for line in lines:
        stripped = line["text"]
        match = re.match(LANDOLT_FOOTNOTE_NUMBER_PATTERN, stripped)
        if match:
            if is_landolt_table_row_text(stripped):
                current_number = None
                continue
            current_number = match.group("num")
            regions[current_number] = {"text": match.group("text"), "bbox": line["bbox"]}
        elif current_number and stripped:
            if re.match(LANDOLT_ROW_NUMBER_PATTERN, stripped):
                current_number = None
            else:
                regions[current_number]["text"] += " " + stripped
                regions[current_number]["bbox"] = union_bbox(regions[current_number]["bbox"], line["bbox"])
    return regions


def union_bbox(left: list[float], right: list[float]) -> list[float]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def render_pdfium_page(pdfium_doc: Any, page_number: int, scale: float) -> Any:
    bitmap = pdfium_doc[page_number - 1].render(scale=scale)
    return bitmap.to_pil().convert("RGB")


def clear_landolt_crop_dir(crop_dir: Path) -> None:
    for path in crop_dir.glob("table*_substance*_p*_landolt-nitrogen-table-*-pdf.png"):
        if path.is_file():
            path.unlink()


def write_landolt_review_crop(
    rendered_page: Any,
    page_width: float,
    region: dict[str, Any],
    entry: dict[str, Any],
    crop_dir: Path,
    scale: float,
) -> str:
    from PIL import Image

    crop_dir.mkdir(parents=True, exist_ok=True)
    boxes = [region["table_bbox"]]
    if region.get("footnote_bbox"):
        boxes.append(region["footnote_bbox"])
    crops = [crop_rendered_bbox(rendered_page, box, scale, margin_points=8.0) for box in boxes]
    gutter = max(18, int(6 * scale))
    width = max(crop.width for crop in crops)
    height = sum(crop.height for crop in crops) + gutter * (len(crops) - 1)
    combined = Image.new("RGB", (width, height), "white")
    y = 0
    for crop in crops:
        combined.paste(crop, (0, y))
        y += crop.height + gutter
    filename = (
        f"table{entry['table_number']}_substance{entry['substance_number']}_"
        f"p{entry['source_page']}_{slug(entry['source_id'])}.png"
    )
    output_path = crop_dir / filename
    combined.save(output_path)
    return str(output_path.relative_to(PROJECT)).replace("\\", "/")


def crop_rendered_bbox(rendered_page: Any, bbox: list[float], scale: float, margin_points: float) -> Any:
    left = max(0, int((bbox[0] - margin_points) * scale))
    upper = max(0, int((bbox[1] - margin_points) * scale))
    right = min(rendered_page.width, int((bbox[2] + margin_points) * scale))
    lower = min(rendered_page.height, int((bbox[3] + margin_points) * scale))
    return rendered_page.crop((left, upper, right, lower))


def landolt_review_issue_flags(
    entry: dict[str, Any], region: dict[str, Any] | None, reference_codes: set[str]
) -> list[str]:
    flags: list[str] = []
    if not region:
        flags.append("no_visual_region")
    if not entry.get("formula_raw"):
        flags.append("missing_formula")
    elif re.search(r"[?&]| [A-Z]?\d", str(entry["formula_raw"])):
        flags.append("formula_ocr_uncertain")
    if not entry.get("substance_name"):
        flags.append("missing_substance_name")
    if not entry.get("cas_registry_number"):
        flags.append("missing_cas")
    if not entry.get("frequencies_raw"):
        flags.append("missing_frequency")
    if not entry.get("qcc_original"):
        flags.append("missing_qcc")
    if not entry.get("eta_original"):
        flags.append("missing_eta")
    if entry.get("table_number") == "9" and entry.get("reference_code") not in reference_codes:
        flags.append("table9_reference_code_unresolved")
    return flags


def landolt_review_priority(issue_flags: list[str]) -> int:
    high_priority = {"no_visual_region", "missing_frequency", "table9_reference_code_unresolved"}
    medium_priority = {"formula_ocr_uncertain", "missing_formula", "missing_substance_name"}
    if any(flag in high_priority for flag in issue_flags):
        return 1
    if any(flag in medium_priority for flag in issue_flags):
        return 2
    return 3


def recreate_database(db_path: Path, state: BuildState) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        insert_many(conn, "sources", state.sources.values())
        insert_many(conn, "compounds", state.compounds.values())
        insert_many(
            conn,
            "compound_aliases",
            [{"compound_id": compound_id, "alias": alias} for compound_id, alias in state.aliases],
        )
        insert_many(conn, "samples", state.samples.values())
        insert_many(conn, "sites", state.sites.values())
        insert_many(conn, "lines", state.lines.values())
        insert_many(conn, "literature_references", state.literature_references.values())
        insert_many(conn, "reference_links", state.reference_links.values())
        insert_many(conn, "nqr_transition_equations", state.nqr_transition_equations.values())
        insert_many(conn, "landolt_column_definitions", state.landolt_column_definitions.values())
        insert_many(conn, "landolt_page_extracts", state.landolt_page_extracts.values())
        insert_many(conn, "landolt_compound_entries", state.landolt_compound_entries.values())
        insert_many(conn, "landolt_reference_codes", state.landolt_reference_codes.values())
        insert_many(conn, "landolt_measurement_sets", state.landolt_measurement_sets.values())
        insert_many(conn, "landolt_frequency_records", state.landolt_frequency_records.values())
        insert_many(conn, "landolt_qcc_eta_records", state.landolt_qcc_eta_records.values())
        insert_many(conn, "landolt_review_queue", state.landolt_review_queue.values())


def insert_many(conn: sqlite3.Connection, table: str, rows: Any) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(columns)
    conn.executemany(
        f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        [[row.get(column) for column in columns] for row in rows],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_normalized(state: BuildState) -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(NORMALIZED_DIR / "sources.jsonl", sorted(state.sources.values(), key=lambda r: r["id"]))
    write_jsonl(NORMALIZED_DIR / "compounds.jsonl", sorted(state.compounds.values(), key=lambda r: r["id"]))
    write_jsonl(NORMALIZED_DIR / "samples.jsonl", sorted(state.samples.values(), key=lambda r: r["id"]))
    write_jsonl(NORMALIZED_DIR / "sites.jsonl", sorted(state.sites.values(), key=lambda r: r["id"]))
    write_jsonl(NORMALIZED_DIR / "lines.jsonl", sorted(state.lines.values(), key=lambda r: r["id"]))
    write_jsonl(
        NORMALIZED_DIR / "literature_references.jsonl",
        sorted(state.literature_references.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "reference_links.jsonl",
        sorted(state.reference_links.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "nqr_transition_equations.jsonl",
        sorted(state.nqr_transition_equations.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_column_definitions.jsonl",
        sorted(state.landolt_column_definitions.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_page_extracts.jsonl",
        sorted(state.landolt_page_extracts.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_compound_entries.jsonl",
        sorted(state.landolt_compound_entries.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_reference_codes.jsonl",
        sorted(state.landolt_reference_codes.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_measurement_sets.jsonl",
        sorted(state.landolt_measurement_sets.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_frequency_records.jsonl",
        sorted(state.landolt_frequency_records.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_qcc_eta_records.jsonl",
        sorted(state.landolt_qcc_eta_records.values(), key=lambda r: r["id"]),
    )
    write_jsonl(
        NORMALIZED_DIR / "landolt_review_queue.jsonl",
        sorted(state.landolt_review_queue.values(), key=lambda r: (r["priority"], r["id"])),
    )

    references_by_line: dict[str, list[dict[str, Any]]] = {}
    references_by_compound: dict[str, list[dict[str, Any]]] = {}
    for link in state.reference_links.values():
        reference = state.literature_references.get(link["reference_id"])
        if not reference:
            continue
        compact = {
            "id": reference["id"],
            "citation_text": reference["citation_text"],
            "reference_type": reference["reference_type"],
            "year": reference["year"],
            "source_page": reference["source_page"],
            "link_type": link["link_type"],
            "note": link["note"],
        }
        if link["line_id"]:
            references_by_line.setdefault(link["line_id"], []).append(compact)
        elif link["compound_id"]:
            references_by_compound.setdefault(link["compound_id"], []).append(compact)

    records = []
    for line in sorted(state.lines.values(), key=lambda r: (r["frequency_khz"] or 0, r["id"])):
        site = state.sites[line["site_id"]]
        sample = state.samples[site["sample_id"]]
        compound = state.compounds[sample["compound_id"]]
        source = state.sources[line["source_id"]]
        refs = references_by_line.get(line["id"], references_by_compound.get(compound["id"], []))
        records.append(
            {
                "line_id": line["id"],
                "compound": {
                    "id": compound["id"],
                    "name": compound["canonical_name"],
                    "formula": compound["formula"],
                    "conventional_formula": compound["conventional_formula"],
                    "category": compound["category"],
                },
                "sample": {
                    "id": sample["id"],
                    "label": sample["label"],
                    "form": sample["form"],
                    "phase": sample["phase"],
                    "temperature_k": sample["temperature_k"],
                },
                "site": {
                    "id": site["id"],
                    "site_number": site["site_number"],
                    "isotope": site["isotope"],
                    "site_label": site["site_label"],
                    "weight_percent": site["weight_percent"],
                    "qcc_khz": site["qcc_khz"],
                    "eta": site["eta"],
                },
                "line": {
                    "frequency_khz": line["frequency_khz"],
                    "frequency_original": line["frequency_original"],
                    "t1_s": line["t1_s"],
                    "t1_original": line["t1_original"],
                    "t2_s": line["t2_s"],
                    "t2_original": line["t2_original"],
                    "t2_star_s": line["t2_star_s"],
                    "t2_star_original": line["t2_star_original"],
                    "fwhm_khz": line["fwhm_khz"],
                    "line_width_khz": line["line_width_khz"],
                    "line_width_original": line["line_width_original"],
                    "dnu_dt_khz_per_c": line["dnu_dt_khz_per_c"],
                    "dnu_dt_original": line["dnu_dt_original"],
                    "polarization_factor": line["polarization_factor"],
                    "polarization_factor_original": line["polarization_factor_original"],
                    "temperature_k": line["temperature_k"],
                    "form": line["form"],
                },
                "source": {
                    "id": source["id"],
                    "title": source["title"],
                    "source_type": source["source_type"],
                    "relative_path": source["relative_path"],
                },
                "references": sorted(refs, key=lambda item: item["id"]),
                "original_record": json.loads(line["original_record"]),
            }
        )
    write_jsonl(NORMALIZED_DIR / "line_records.jsonl", records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwru-dir", type=Path, default=DEFAULT_CWRU_DIR)
    parser.add_argument("--nrl-dir", type=Path, default=DEFAULT_NRL_DIR)
    parser.add_argument("--kcl-dir", type=Path, default=DEFAULT_KCL_DIR)
    parser.add_argument("--landolt-dir", type=Path, default=DEFAULT_LANDOLT_DIR)
    parser.add_argument("--landolt-crop-dir", type=Path, default=LANDOLT_CROP_DIR)
    parser.add_argument("--skip-landolt-crops", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = build_state(args.cwru_dir)
    import_nrl_summaries(state, args.nrl_dir)
    import_nrl_references(state)
    import_kcl_notes(state, args.kcl_dir)
    import_landolt_material(state, args.landolt_dir)
    build_landolt_independent_value_records(state)
    build_landolt_review_queue(
        state,
        args.landolt_dir,
        args.landolt_crop_dir,
        generate_crops=not args.skip_landolt_crops,
    )
    promote_accepted_landolt_reviews(state)
    write_normalized(state)
    recreate_database(args.output, state)

    print(f"sources={len(state.sources)}")
    print(f"compounds={len(state.compounds)}")
    print(f"samples={len(state.samples)}")
    print(f"sites={len(state.sites)}")
    print(f"lines={len(state.lines)}")
    print(f"literature_references={len(state.literature_references)}")
    print(f"reference_links={len(state.reference_links)}")
    print(f"nqr_transition_equations={len(state.nqr_transition_equations)}")
    print(f"landolt_compound_entries={len(state.landolt_compound_entries)}")
    print(f"landolt_reference_codes={len(state.landolt_reference_codes)}")
    print(f"landolt_measurement_sets={len(state.landolt_measurement_sets)}")
    print(f"landolt_frequency_records={len(state.landolt_frequency_records)}")
    print(f"landolt_qcc_eta_records={len(state.landolt_qcc_eta_records)}")
    print(f"landolt_review_queue={len(state.landolt_review_queue)}")
    print(f"sqlite={args.output}")
    print(f"jsonl={NORMALIZED_DIR / 'line_records.jsonl'}")


if __name__ == "__main__":
    main()
