"""Local GUI server for reviewing staged Landolt NQR entries."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
ROOT = PROJECT.parent
DB_PATH = PROJECT / "data" / "exports" / "nqr.sqlite"
DECISIONS_PATH = PROJECT / "data" / "review" / "landolt_review_decisions.jsonl"
CROP_DIR = PROJECT / "data" / "review" / "landolt_crops"
STATIC_DIR = APP_DIR / "static"

STATUSES = {"unreviewed", "accepted", "needs_manual_fix", "rejected"}
EDITABLE_FIELDS = {
    "formula_raw",
    "nucleus",
    "method",
    "temperature_original",
    "frequencies_raw",
    "qcc_original",
    "eta_original",
    "reference_code",
    "substance_name",
    "cas_registry_number",
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw else {}


def latest_decisions() -> dict[str, dict]:
    decisions: dict[str, dict] = {}
    if not DECISIONS_PATH.exists():
        return decisions
    with DECISIONS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            decisions[record["review_id"]] = record
    return decisions


def append_decision(record: dict) -> None:
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def review_rows(query: dict[str, list[str]]) -> list[dict]:
    status = first_query_value(query, "status")
    priority = first_query_value(query, "priority")
    search = first_query_value(query, "q")
    clauses: list[str] = []
    params: list[str | int] = []
    if search:
        like = f"%{search}%"
        clauses.append(
            "(e.substance_number LIKE ? OR e.substance_name LIKE ? OR e.formula_raw LIKE ? "
            "OR e.cas_registry_number LIKE ? OR e.reference_code LIKE ?)"
        )
        params.extend([like, like, like, like, like])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT
            q.id, q.entry_id, q.status, q.priority, q.issue_flags_json,
            q.crop_relative_path, q.source_id, q.source_page, q.reviewer_notes,
            q.updated_at, e.table_number, e.substance_number, e.formula_raw,
            e.reference_code, e.substance_name, e.cas_registry_number,
            e.frequencies_raw, e.qcc_original, e.eta_original
        FROM landolt_review_queue q
        JOIN landolt_compound_entries e ON e.id = q.entry_id
        {where}
        ORDER BY q.priority ASC, q.status DESC, CAST(e.table_number AS INTEGER), CAST(e.substance_number AS INTEGER)
    """
    decisions = latest_decisions()
    with connect() as conn:
        rows = [row_to_payload(dict(row), decisions) for row in conn.execute(sql, params)]
    if status and status != "all":
        rows = [row for row in rows if row["status"] == status]
    if priority and priority != "all":
        rows = [row for row in rows if row["priority"] == int(priority)]
    return rows


def review_counts() -> dict:
    decisions = latest_decisions()
    with connect() as conn:
        rows = [
            row_to_payload(dict(row), decisions)
            for row in conn.execute("SELECT * FROM landolt_review_queue")
        ]
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        priority = str(row["priority"])
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    return {
        "status": status_counts,
        "priority": priority_counts,
    }


def review_item(review_id: str) -> dict | None:
    sql = """
        SELECT
            q.*, e.table_number, e.substance_number, e.formula_raw, e.nucleus,
            e.method, e.temperature_original, e.frequencies_raw, e.qcc_original,
            e.eta_original, e.reference_code, e.remark_flag, e.substance_name,
            e.cas_registry_number, e.raw_table_text, e.raw_footnote_text,
            e.extraction_confidence, e.notes
        FROM landolt_review_queue q
        JOIN landolt_compound_entries e ON e.id = q.entry_id
        WHERE q.id = ?
    """
    decisions = latest_decisions()
    with connect() as conn:
        row = conn.execute(sql, [review_id]).fetchone()
        if not row:
            return None
        payload = row_to_payload(dict(row), decisions)
        decision = decisions.get(payload["id"])
        payload["measurement_sets"] = reviewed_measurement_sets(conn, payload, decision)
        payload["frequency_records"] = flatten_measurement_set_records(payload["measurement_sets"], "frequency_records")
        payload["qcc_eta_records"] = flatten_measurement_set_records(payload["measurement_sets"], "qcc_eta_records")
        payload["consistency"] = consistency_flag(conn, payload["entry_id"])
        return payload


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            [name],
        ).fetchone()
        is not None
    )


def consistency_flag(conn: sqlite3.Connection, entry_id: str) -> dict | None:
    """Return the simulator consistency diagnostic for one Landolt entry.

    Produced by the sibling ``mr_integration`` project as an optional overlay;
    the review GUI omits the banner when the table is absent.
    """

    if not table_exists(conn, "landolt_consistency_flags"):
        return None
    row = conn.execute(
        "SELECT * FROM landolt_consistency_flags WHERE entry_id = ?",
        [entry_id],
    ).fetchone()
    return dict(row) if row else None


def row_to_payload(row: dict, decisions: dict[str, dict]) -> dict:
    row["issue_flags"] = json.loads(row.pop("issue_flags_json") or "[]")
    if row.get("crop_bbox_json"):
        row["crop_bbox"] = json.loads(row.pop("crop_bbox_json"))
    decision = decisions.get(row["id"])
    row["decision"] = decision
    if decision:
        row["status"] = decision.get("status", row["status"])
        row["reviewer_notes"] = decision.get("reviewer_notes", row.get("reviewer_notes"))
        for field, value in decision.get("field_edits", {}).items():
            row[field] = value
    if row.get("formula_raw"):
        row["formula_raw"] = normalize_formula_ocr(str(row["formula_raw"]))
    if row.get("cas_registry_number"):
        row["cas_registry_number"] = normalize_cas_ocr(str(row["cas_registry_number"]))
    return row


def save_review(review_id: str, payload: dict) -> dict:
    status = payload.get("status", "unreviewed")
    if status not in STATUSES:
        raise ValueError(f"Unsupported status: {status}")
    field_edits = {
        key: clean_edit_value(value)
        for key, value in payload.get("field_edits", {}).items()
        if key in EDITABLE_FIELDS
    }
    measurement_sets = clean_measurement_sets(payload.get("measurement_sets", []))
    if not measurement_sets and (
        payload.get("frequency_records") or payload.get("qcc_eta_records")
    ):
        measurement_sets = [
            {
                "set_index": 1,
                "method": None,
                "method_description": None,
                "temperature_original": None,
                "reference_code": None,
                "remark_flag": None,
                "raw_set_text": None,
                "notes": None,
                "frequency_records": clean_frequency_records(payload.get("frequency_records", [])),
                "qcc_eta_records": clean_qcc_eta_records(payload.get("qcc_eta_records", [])),
            }
        ]
    reviewer_notes = clean_edit_value(payload.get("reviewer_notes"))
    updated_at = utc_now()
    record = {
        "review_id": review_id,
        "status": status,
        "field_edits": field_edits,
        "measurement_sets": measurement_sets,
        "frequency_records": flatten_measurement_set_records(measurement_sets, "frequency_records"),
        "qcc_eta_records": flatten_measurement_set_records(measurement_sets, "qcc_eta_records"),
        "reviewer_notes": reviewer_notes,
        "updated_at": updated_at,
    }
    append_decision(record)
    with connect() as conn:
        conn.execute(
            """
            UPDATE landolt_review_queue
            SET status = ?, reviewer_notes = ?, updated_at = ?
            WHERE id = ?
            """,
            [status, reviewer_notes, updated_at, review_id],
        )
        conn.commit()
    item = review_item(review_id)
    if not item:
        raise ValueError("Review row not found after update")
    return item


def reviewed_measurement_sets(conn: sqlite3.Connection, row: dict, decision: dict | None) -> list[dict]:
    if decision and "measurement_sets" in decision:
        return clean_measurement_sets(decision["measurement_sets"])
    sets = [
        dict(record)
        for record in conn.execute(
            """
            SELECT id, set_index, method, method_description, temperature_original,
                   reference_code, remark_flag, raw_set_text, notes
            FROM landolt_measurement_sets
            WHERE entry_id = ?
            ORDER BY set_index
            """,
            [row["entry_id"]],
        )
    ]
    for measurement_set in sets:
        measurement_set["frequency_records"] = clean_frequency_records(
            [
                dict(record)
                for record in conn.execute(
                    """
                    SELECT sequence_index, frequency_original, notes
                    FROM landolt_frequency_records
                    WHERE measurement_set_id = ?
                    ORDER BY sequence_index
                    """,
                    [measurement_set["id"]],
                )
            ]
        )
        measurement_set["qcc_eta_records"] = clean_qcc_eta_records(
            [
                dict(record)
                for record in conn.execute(
                    """
                    SELECT sequence_index, qcc_original, eta_original, notes
                    FROM landolt_qcc_eta_records
                    WHERE measurement_set_id = ?
                    ORDER BY sequence_index
                    """,
                    [measurement_set["id"]],
                )
            ]
        )
    if sets:
        if decision and "measurement_sets" not in decision and len(sets) == 1:
            if "frequency_records" in decision:
                sets[0]["frequency_records"] = clean_frequency_records(decision["frequency_records"])
            if "qcc_eta_records" in decision:
                sets[0]["qcc_eta_records"] = clean_qcc_eta_records(decision["qcc_eta_records"])
        return clean_measurement_sets(sets)
    return [
        {
            "set_index": 1,
            "method": row.get("method"),
            "method_description": None,
            "temperature_original": row.get("temperature_original"),
            "reference_code": row.get("reference_code"),
            "remark_flag": row.get("remark_flag"),
            "raw_set_text": row.get("raw_table_text"),
            "notes": "Fallback measurement set from legacy entry-level fields.",
            "frequency_records": reviewed_frequency_records(conn, row, decision),
            "qcc_eta_records": reviewed_qcc_eta_records(conn, row, decision),
        }
    ]


def reviewed_frequency_records(conn: sqlite3.Connection, row: dict, decision: dict | None) -> list[dict]:
    if decision and "frequency_records" in decision:
        return decision["frequency_records"]
    records = [
        dict(record)
        for record in conn.execute(
            """
            SELECT sequence_index, frequency_original, notes
            FROM landolt_frequency_records
            WHERE entry_id = ?
            ORDER BY sequence_index
            """,
            [row["entry_id"]],
        )
    ]
    return clean_frequency_records(records)


def reviewed_qcc_eta_records(conn: sqlite3.Connection, row: dict, decision: dict | None) -> list[dict]:
    if decision and "qcc_eta_records" in decision:
        return decision["qcc_eta_records"]
    records = [
        dict(record)
        for record in conn.execute(
            """
            SELECT sequence_index, qcc_original, eta_original, notes
            FROM landolt_qcc_eta_records
            WHERE entry_id = ?
            ORDER BY sequence_index
            """,
            [row["entry_id"]],
        )
    ]
    return clean_qcc_eta_records(records)


def clean_measurement_sets(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict] = []
    for idx, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            continue
        row = {
            "set_index": safe_int(record.get("set_index"), idx),
            "method": clean_edit_value(record.get("method")),
            "method_description": clean_edit_value(record.get("method_description")),
            "temperature_original": clean_edit_value(record.get("temperature_original")),
            "reference_code": clean_edit_value(record.get("reference_code")),
            "remark_flag": clean_edit_value(record.get("remark_flag")),
            "raw_set_text": clean_edit_value(record.get("raw_set_text")),
            "notes": clean_edit_value(record.get("notes")),
            "frequency_records": clean_frequency_records(record.get("frequency_records", [])),
            "qcc_eta_records": clean_qcc_eta_records(record.get("qcc_eta_records", [])),
        }
        if any(
            row.get(field)
            for field in (
                "method",
                "temperature_original",
                "reference_code",
                "remark_flag",
                "raw_set_text",
                "notes",
            )
        ) or row["frequency_records"] or row["qcc_eta_records"]:
            cleaned.append(row)
    return cleaned


def flatten_measurement_set_records(measurement_sets: list[dict], key: str) -> list[dict]:
    records: list[dict] = []
    for measurement_set in measurement_sets:
        for record in measurement_set.get(key, []):
            row = dict(record)
            row["measurement_set_index"] = measurement_set.get("set_index")
            row["measurement_method"] = measurement_set.get("method")
            row["measurement_temperature_original"] = measurement_set.get("temperature_original")
            row["measurement_reference_code"] = measurement_set.get("reference_code")
            records.append(row)
    return records


def clean_frequency_records(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict] = []
    for idx, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            continue
        row = {
            "sequence_index": int(record.get("sequence_index") or idx),
            "frequency_original": clean_edit_value(record.get("frequency_original")),
            "notes": clean_edit_value(record.get("notes")),
        }
        if row["frequency_original"] or row["notes"]:
            cleaned.append(row)
    return cleaned


def clean_qcc_eta_records(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict] = []
    for idx, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            continue
        row = {
            "sequence_index": int(record.get("sequence_index") or idx),
            "qcc_original": clean_edit_value(record.get("qcc_original")),
            "eta_original": clean_edit_value(record.get("eta_original")),
            "notes": clean_edit_value(record.get("notes")),
        }
        if row["qcc_original"] or row["eta_original"] or row["notes"]:
            cleaned.append(row)
    return cleaned


def safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def split_source_list(value: object) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in re_split_source_list(str(value)) if part.strip()]


def re_split_source_list(value: str) -> list[str]:
    return re.split(r"[;\n,]+", value)


def normalize_cas_ocr(value: str) -> str:
    match = re.fullmatch(r"(\d{2,7}-\d{2}-\d)(?:1)?", value.strip())
    return match.group(1) if match else value.strip()


def normalize_formula_ocr(value: str) -> str:
    text = normalize_formula_token(value.replace(" ", "").replace("Q", "O"))
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"C1(?=[A-Z]|$)", "Cl", text)
        text = normalize_formula_token(text)
    return text


def normalize_formula_token(value: str) -> str:
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


def clean_edit_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def safe_crop_path(relative: str) -> Path:
    rel = Path(unquote(relative))
    candidate = (PROJECT / rel).resolve()
    crop_root = CROP_DIR.resolve()
    if crop_root not in candidate.parents and candidate != crop_root:
        raise ValueError("Crop path is outside the review crop directory")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def safe_static_path(relative: str) -> Path:
    rel = Path(unquote(relative))
    candidate = (STATIC_DIR / rel).resolve()
    static_root = STATIC_DIR.resolve()
    if static_root not in candidate.parents and candidate != static_root:
        raise ValueError("Static path is outside the app directory")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "NQRReview/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_static(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self.send_static(safe_static_path(parsed.path.removeprefix("/static/")))
            elif parsed.path == "/api/queue":
                rows = review_rows(parse_qs(parsed.query))
                self.send_json({"rows": rows, "counts": review_counts()})
            elif parsed.path.startswith("/api/item/"):
                item = review_item(unquote(parsed.path.removeprefix("/api/item/")))
                if not item:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(item)
            elif parsed.path == "/api/counts":
                self.send_json(review_counts())
            elif parsed.path.startswith("/crops/"):
                self.send_static(safe_crop_path(parsed.path.removeprefix("/crops/")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/review/"):
                review_id = unquote(parsed.path.removeprefix("/api/review/"))
                self.send_json(save_review(review_id, read_json_body(self)))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
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
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Landolt review GUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
