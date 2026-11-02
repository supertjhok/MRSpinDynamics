"""The static Pages explorer must answer exactly as the explorer server does."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "app"))

import explorer_server as server  # noqa: E402
from build_static_explorer import build, compound_file_name  # noqa: E402

# Queries covering text, every filter, frequency bounds and their combination.
QUERIES = [
    {},
    {"q": "nitrate"},
    {"q": "NaNO2"},
    {"q": "CAS"},
    {"q": "zz-no-such-compound"},
    {"isotope": "14N"},
    {"isotope": "35Cl"},
    {"category": "explosives"},
    {"source_type": "landolt_bornstein_pdf"},
    {"freq_min": "3000"},
    {"freq_max": "800"},
    {"freq_min": "600", "freq_max": "900", "isotope": "14N"},
    {"isotope": "14N", "limit": "300"},
    {"q": "amine", "freq_max": "3500", "limit": "5"},
]

_NODE_SEARCH = """
const { searchRows } = require(process.argv[1]);
const index = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const queries = JSON.parse(require("fs").readFileSync(0, "utf8"));
const results = queries.map(query => searchRows(index, new URLSearchParams(query)).rows.map(row => row.id));
process.stdout.write(JSON.stringify(results));
"""


class StaticExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.site = cls.tmp / "explorer"
        cls.summary = build(cls.site)
        cls.index = json.loads((cls.site / "data" / "index.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_site_files_and_relative_assets(self) -> None:
        for name in ("index.html", "app.js", "static-api.js", "styles.css"):
            self.assertTrue((self.site / name).is_file(), name)
        html = (self.site / "index.html").read_text(encoding="utf-8")
        # Pages serves the site under a sub-path, so nothing may be root-relative.
        self.assertNotIn('"/static/', html)
        self.assertLess(html.index("static-api.js"), html.index('src="app.js"'))

    def test_every_compound_has_a_detail_file_matching_the_server(self) -> None:
        with server.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        self.assertEqual(len(self.index), count)
        self.assertEqual(self.summary["compounds"], count)
        files = {entry["file"] for entry in self.index}
        self.assertEqual(len(files), count)
        for entry in self.index[:: max(1, count // 12)]:
            detail = json.loads(
                (self.site / "data" / "compounds" / entry["file"]).read_text(encoding="utf-8")
            )
            expected = server.compound_detail(entry["row"]["id"])
            self.assertEqual(detail["id"], entry["row"]["id"])
            self.assertEqual(detail["spectrum"], expected["spectrum"])
            self.assertNotIn("original_record", json.dumps(detail))

    def test_stats_and_options_match_the_server(self) -> None:
        data = self.site / "data"
        self.assertEqual(json.loads((data / "stats.json").read_text(encoding="utf-8")), server.explorer_stats())
        self.assertEqual(json.loads((data / "options.json").read_text(encoding="utf-8")), server.source_options())

    def test_review_tables_are_not_published(self) -> None:
        published = " ".join(path.name for path in self.site.rglob("*"))
        self.assertNotIn("landolt_review", published)
        text = (self.site / "data" / "index.json").read_text(encoding="utf-8")
        self.assertNotIn("landolt_page_extracts", text)

    def test_file_names_are_unique_and_url_safe(self) -> None:
        used: set[str] = set()
        names = [compound_file_name(value, used) for value in ("a:b", "a/b", "A:B", "", "x y")]
        self.assertEqual(len(set(name.lower() for name in names)), len(names))
        for name in names:
            self.assertRegex(name, r"^[A-Za-z0-9._-]+\.json$")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_javascript_search_matches_the_server_search(self) -> None:
        completed = subprocess.run(
            [
                "node",
                "-e",
                _NODE_SEARCH,
                str(self.site / "static-api.js"),
                str(self.site / "data" / "index.json"),
            ],
            input=json.dumps(QUERIES),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        results = json.loads(completed.stdout)
        for query, static_ids in zip(QUERIES, results):
            expected = server.search_compounds({key: [value] for key, value in query.items()})
            with self.subTest(query=query):
                self.assertEqual(static_ids, [row["id"] for row in expected["rows"]])


if __name__ == "__main__":
    unittest.main()
