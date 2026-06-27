"""Local GUI server for exploring the canonical NQR database."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
DB_PATH = PROJECT / "data" / "exports" / "nqr.sqlite"
STATIC_DIR = APP_DIR / "explorer_static"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def row_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def split_group_concat(value: str | None) -> list[str]:
    if not value:
        return []
    return sorted({item for item in value.split("|") if item})


def explorer_stats() -> dict:
    with connect() as conn:
        scalar = {
            name: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for name, table in [
                ("compounds", "compounds"),
                ("samples", "samples"),
                ("sites", "sites"),
                ("lines", "lines"),
                ("references", "literature_references"),
                ("sources", "sources"),
            ]
        }
        categories = [
            row_dict(row)
            for row in conn.execute(
                """
                SELECT COALESCE(category, 'uncategorized') AS category, COUNT(*) AS count
                FROM compounds
                GROUP BY COALESCE(category, 'uncategorized')
                ORDER BY count DESC, category
                """
            )
        ]
        isotopes = [
            row_dict(row)
            for row in conn.execute(
                """
                SELECT COALESCE(isotope, 'unknown') AS isotope, COUNT(DISTINCT sites.id) AS count
                FROM sites
                GROUP BY COALESCE(isotope, 'unknown')
                ORDER BY count DESC, isotope
                """
            )
        ]
        source_types = [
            row_dict(row)
            for row in conn.execute(
                """
                SELECT sources.source_type, COUNT(lines.id) AS line_count
                FROM sources
                LEFT JOIN lines ON lines.source_id = sources.id
                GROUP BY sources.source_type
                ORDER BY line_count DESC, sources.source_type
                """
            )
        ]
    return {
        "counts": scalar,
        "categories": categories,
        "isotopes": isotopes,
        "source_types": source_types,
    }


def source_options() -> dict:
    with connect() as conn:
        categories = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT category FROM compounds WHERE category IS NOT NULL ORDER BY category"
            )
        ]
        isotopes = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT isotope FROM sites WHERE isotope IS NOT NULL ORDER BY isotope"
            )
        ]
        source_types = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source_type FROM sources ORDER BY source_type"
            )
        ]
    return {
        "categories": categories,
        "isotopes": isotopes,
        "source_types": source_types,
    }


def search_compounds(query: dict[str, list[str]]) -> dict:
    text = first_query_value(query, "q")
    category = first_query_value(query, "category")
    isotope = first_query_value(query, "isotope")
    source_type = first_query_value(query, "source_type")
    freq_min = safe_float(first_query_value(query, "freq_min"))
    freq_max = safe_float(first_query_value(query, "freq_max"))
    limit = int(first_query_value(query, "limit") or "80")
    limit = max(1, min(limit, 300))

    clauses: list[str] = []
    params: list[object] = []
    if text:
        like = f"%{text}%"
        clauses.append(
            """
            (
                c.canonical_name LIKE ?
                OR c.formula LIKE ?
                OR c.conventional_formula LIKE ?
                OR c.notes LIKE ?
                OR EXISTS (
                    SELECT 1 FROM compound_aliases ca
                    WHERE ca.compound_id = c.id AND ca.alias LIKE ?
                )
            )
            """
        )
        params.extend([like, like, like, like, like])
    if category:
        clauses.append("c.category = ?")
        params.append(category)
    if isotope:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM samples fs
                JOIN sites fst ON fst.sample_id = fs.id
                WHERE fs.compound_id = c.id AND fst.isotope = ?
            )
            """
        )
        params.append(isotope)
    if source_type:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM samples ss
                JOIN sites sst ON sst.sample_id = ss.id
                LEFT JOIN lines sl ON sl.site_id = sst.id
                LEFT JOIN sources ssrc ON ssrc.id = COALESCE(sl.source_id, sst.source_id)
                WHERE ss.compound_id = c.id AND ssrc.source_type = ?
            )
            """
        )
        params.append(source_type)
    if freq_min is not None or freq_max is not None:
        frequency_terms = ["rfs.compound_id = c.id"]
        if freq_min is not None:
            frequency_terms.append("rfl.frequency_khz >= ?")
            params.append(freq_min)
        if freq_max is not None:
            frequency_terms.append("rfl.frequency_khz <= ?")
            params.append(freq_max)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM samples rfs
                JOIN sites rfst ON rfst.sample_id = rfs.id
                JOIN lines rfl ON rfl.site_id = rfst.id
                WHERE {" AND ".join(frequency_terms)}
            )
            """
        )

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT
            c.id,
            c.canonical_name,
            c.formula,
            c.conventional_formula,
            c.category,
            COUNT(DISTINCT samples.id) AS sample_count,
            COUNT(DISTINCT sites.id) AS site_count,
            COUNT(DISTINCT lines.id) AS line_count,
            MIN(lines.frequency_khz) AS min_frequency_khz,
            MAX(lines.frequency_khz) AS max_frequency_khz,
            GROUP_CONCAT(DISTINCT sites.isotope) AS isotopes_raw,
            GROUP_CONCAT(DISTINCT sources.source_type) AS source_types_raw
        FROM compounds c
        LEFT JOIN samples ON samples.compound_id = c.id
        LEFT JOIN sites ON sites.sample_id = samples.id
        LEFT JOIN lines ON lines.site_id = sites.id
        LEFT JOIN sources ON sources.id = COALESCE(lines.source_id, sites.source_id)
        {where}
        GROUP BY c.id
        ORDER BY line_count DESC, c.canonical_name
        LIMIT ?
    """
    params.append(limit)
    with connect() as conn:
        rows = []
        for row in conn.execute(sql, params):
            payload = row_dict(row)
            payload["isotopes"] = split_sqlite_list(payload.pop("isotopes_raw"))
            payload["source_types"] = split_sqlite_list(payload.pop("source_types_raw"))
            payload["structure"] = structure_payload(payload, aliases=[])
            rows.append(payload)
    return {"rows": rows, "count": len(rows)}


def split_sqlite_list(value: str | None) -> list[str]:
    if not value:
        return []
    return sorted({item for item in value.split(",") if item})


def compound_detail(compound_id: str) -> dict | None:
    with connect() as conn:
        compound = conn.execute(
            "SELECT * FROM compounds WHERE id = ?",
            [compound_id],
        ).fetchone()
        if not compound:
            return None
        aliases = [
            row[0]
            for row in conn.execute(
                "SELECT alias FROM compound_aliases WHERE compound_id = ? ORDER BY alias",
                [compound_id],
            )
        ]
        samples = []
        for sample_row in conn.execute(
            "SELECT * FROM samples WHERE compound_id = ? ORDER BY temperature_k, label",
            [compound_id],
        ):
            sample = row_dict(sample_row)
            sample["measurement"] = measurement_payload(sample)
            sample["sites"] = []
            for site_row in conn.execute(
                """
                SELECT sites.*, sources.title AS source_title, sources.source_type
                FROM sites
                LEFT JOIN sources ON sources.id = sites.source_id
                WHERE sites.sample_id = ?
                ORDER BY sites.site_number IS NULL, sites.site_number, sites.site_label
                """,
                [sample["id"]],
            ):
                site = row_dict(site_row)
                site["measurement"] = measurement_payload(site)
                site["lines"] = [
                    line_payload(row_dict(line), sample, site)
                    for line in conn.execute(
                        """
                        SELECT lines.*, sources.title AS source_title, sources.source_type
                        FROM lines
                        LEFT JOIN sources ON sources.id = lines.source_id
                        WHERE lines.site_id = ?
                        ORDER BY lines.frequency_khz
                        """,
                        [site["id"]],
                    )
                ]
                site["references"] = linked_references(conn, compound_id, site["id"], None)
                for line in site["lines"]:
                    line["references"] = linked_references(conn, compound_id, site["id"], line["id"])
                sample["sites"].append(site)
            samples.append(sample)
        references = linked_references(conn, compound_id, None, None)
        sources = [
            row_dict(row)
            for row in conn.execute(
                """
                SELECT DISTINCT sources.*
                FROM samples
                JOIN sites ON sites.sample_id = samples.id
                LEFT JOIN lines ON lines.site_id = sites.id
                JOIN sources ON sources.id = COALESCE(lines.source_id, sites.source_id)
                WHERE samples.compound_id = ?
                ORDER BY sources.source_type, sources.title
                """,
                [compound_id],
            )
        ]
    payload = row_dict(compound)
    payload["aliases"] = aliases
    payload["samples"] = samples
    payload["references"] = references
    payload["sources"] = sources
    payload["structure"] = structure_payload(payload, aliases)
    payload["spectrum"] = spectrum_payload(samples)
    return payload


def linked_references(
    conn: sqlite3.Connection,
    compound_id: str,
    site_id: str | None,
    line_id: str | None,
) -> list[dict]:
    params: list[object] = []
    clauses = []
    if line_id:
        clauses.append("reference_links.line_id = ?")
        params.append(line_id)
    if site_id:
        clauses.append("reference_links.site_id = ?")
        params.append(site_id)
    if not line_id and not site_id:
        clauses.append("reference_links.compound_id = ?")
        params.append(compound_id)
    if not clauses:
        return []
    sql = f"""
        SELECT DISTINCT
            literature_references.*,
            reference_links.link_type,
            reference_links.note
        FROM reference_links
        JOIN literature_references ON literature_references.id = reference_links.reference_id
        WHERE {" OR ".join(clauses)}
        ORDER BY literature_references.year, literature_references.citation_text
    """
    return [row_dict(row) for row in conn.execute(sql, params)]


def spectrum_payload(samples: list[dict]) -> list[dict]:
    points: list[dict] = []
    for sample in samples:
        for site in sample["sites"]:
            for line in site["lines"]:
                frequency = line.get("frequency_khz")
                if frequency is None:
                    continue
                points.append(
                    {
                        "frequency_khz": frequency,
                        "frequency_original": line.get("frequency_original"),
                        "site_id": site.get("id"),
                        "isotope": site.get("isotope"),
                        "site_label": site.get("site_label"),
                        "sample_label": sample.get("label"),
                        "temperature_k": line.get("temperature_k") or sample.get("temperature_k"),
                        "temperature_original": line.get("measurement", {}).get("temperature_original")
                        or sample.get("measurement", {}).get("temperature_original"),
                        "method": line.get("measurement", {}).get("method")
                        or sample.get("measurement", {}).get("method"),
                        "method_description": line.get("measurement", {}).get("method_description")
                        or sample.get("measurement", {}).get("method_description"),
                        "source_type": line.get("source_type") or site.get("source_type"),
                    }
                )
    return sorted(points, key=lambda item: item["frequency_khz"])


def line_payload(line: dict, sample: dict, site: dict) -> dict:
    line["measurement"] = measurement_payload(line, sample, site)
    return line


def measurement_payload(record: dict, sample: dict | None = None, site: dict | None = None) -> dict:
    original = parse_original_record(record.get("original_record"))
    measurement_set = original.get("measurement_set") if isinstance(original.get("measurement_set"), dict) else {}
    method = measurement_set.get("method")
    method_description = measurement_set.get("method_description")
    temperature_original = measurement_set.get("temperature_original")
    notes = record.get("notes") or ""
    if not method:
        method_match = re.search(r"Method\s+([A-Z]):\s*([^.;]+)", notes)
        if method_match:
            method = method_match.group(1)
            method_description = method_description or method_match.group(2)
    if not temperature_original:
        temperature_match = re.search(r"Temperature original:\s*([^.;]+)", notes)
        if temperature_match:
            temperature_original = temperature_match.group(1)
    if not temperature_original and record.get("temperature_k") is not None:
        temperature_original = f"{record['temperature_k']:g} K"
    if not temperature_original and sample and sample.get("temperature_k") is not None:
        temperature_original = f"{sample['temperature_k']:g} K"
    return {
        "method": method,
        "method_description": method_description,
        "temperature_original": temperature_original,
        "temperature_k": record.get("temperature_k") or (sample or {}).get("temperature_k"),
        "form": record.get("form") or (sample or {}).get("form"),
        "phase": record.get("phase") or (sample or {}).get("phase"),
        "curation_method": original.get("curation_method"),
        "source_row": original.get("row"),
    }


def parse_original_record(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def structure_payload(compound: dict, aliases: list[str]) -> dict:
    candidates: list[dict[str, str]] = []
    for alias in aliases:
        if re.fullmatch(r"CAS \d{2,7}-\d{2}-\d", alias):
            cas = alias.removeprefix("CAS ")
            candidates.append(pubchem_candidate(cas, "CAS registry number"))
    name = compound.get("canonical_name")
    if name:
        candidates.append(pubchem_candidate(name, "compound name"))
    formula = compound.get("conventional_formula") or compound.get("formula")
    return {
        "formula": formula,
        "candidates": candidates,
        "pubchem_search_url": pubchem_search_url(name or formula or ""),
    }


def pubchem_candidate(value: str, label: str) -> dict[str, str]:
    encoded = quote(value, safe="")
    return {
        "label": label,
        "value": value,
        "image_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/PNG",
        "page_url": f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded}",
    }


def pubchem_search_url(value: str) -> str | None:
    if not value:
        return None
    return f"https://pubchem.ncbi.nlm.nih.gov/#query={quote(value, safe='')}"


def safe_static_path(relative: str) -> Path:
    rel = Path(unquote(relative))
    candidate = (STATIC_DIR / rel).resolve()
    static_root = STATIC_DIR.resolve()
    if static_root not in candidate.parents and candidate != static_root:
        raise ValueError("Static path is outside the explorer directory")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


class ExplorerHandler(BaseHTTPRequestHandler):
    server_version = "NQRExplorer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_static(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self.send_static(safe_static_path(parsed.path.removeprefix("/static/")))
            elif parsed.path == "/api/stats":
                self.send_json(explorer_stats())
            elif parsed.path == "/api/options":
                self.send_json(source_options())
            elif parsed.path == "/api/search":
                self.send_json(search_compounds(parse_qs(parsed.query)))
            elif parsed.path.startswith("/api/compound/"):
                compound_id = unquote(parsed.path.removeprefix("/api/compound/"))
                detail = compound_detail(compound_id)
                if not detail:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(detail)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ExplorerHandler)
    print(f"NQR database explorer: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
