"""Data-path resolution, read-only connections, and reproducible overlays."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mr_integration import (
    measured_lines,
    write_consistency_flags,
    write_landolt_review_flags,
)
from mr_integration import database as database_module
from mr_integration.database import default_database_path, readonly_uri
from mr_integration.flag_export import deterministic_generated_at

_SHIPPED_DB = default_database_path()
_HAS_DB = _SHIPPED_DB.is_file()


class DataPathTests(unittest.TestCase):
    def test_env_var_overrides_default_database(self) -> None:
        with mock.patch.dict(os.environ, {"MR_INTEGRATION_NQR_DB": "/x/y/nqr.sqlite"}):
            self.assertEqual(default_database_path(), Path("/x/y/nqr.sqlite"))

    def test_repo_root_env_var_relocates_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = {"MR_INTEGRATION_REPO_ROOT": directory}
            with mock.patch.dict(os.environ, env):
                os.environ.pop("MR_INTEGRATION_NQR_DB", None)
                expected = Path(directory) / "NQRDatabase" / "data" / "exports"
                self.assertEqual(default_database_path().parent, expected)
                with self.assertRaises(FileNotFoundError) as caught:
                    measured_lines("NaNO2")
        message = str(caught.exception)
        self.assertIn("MR_INTEGRATION_NQR_DB", message)
        self.assertIn("does not bundle data", message)

    def test_explicit_missing_database_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            measured_lines("NaNO2", database_path="no/such/nqr.sqlite")

    def test_readonly_uri_escapes_uri_metacharacters(self) -> None:
        uri = readonly_uri(Path("/tmp/a b#c?d%e/nqr.sqlite"))
        self.assertTrue(uri.startswith("file:"))
        self.assertTrue(uri.endswith("?mode=ro"))
        body = uri[: -len("?mode=ro")]
        for raw in (" ", "#", "?"):
            self.assertNotIn(raw, body)


@unittest.skipUnless(_HAS_DB, "NQR SQLite export absent")
class ConnectionTests(unittest.TestCase):
    def test_reads_database_under_awkward_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            awkward = Path(directory) / "odd dir #1 %20 ?x"
            awkward.mkdir()
            db = awkward / "nqr.sqlite"
            shutil.copy(_SHIPPED_DB, db)
            lines = measured_lines("NaNO2", isotope="14N", database_path=db)
        self.assertEqual(len(lines), 3)

    def test_connections_are_closed_after_queries(self) -> None:
        opened: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def recording_connect(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with mock.patch.object(database_module.sqlite3, "connect", recording_connect):
            measured_lines("NaNO2")
            database_module.sites_with_parameters()
        self.assertEqual(len(opened), 2)
        for connection in opened:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


class DeterministicStampTests(unittest.TestCase):
    def test_stamp_is_a_content_fingerprint(self) -> None:
        rows = [{"site_id": "a", "value": 1.0, "generated_at": "ignored"}]
        with mock.patch.dict(os.environ):
            os.environ.pop("SOURCE_DATE_EPOCH", None)
            first = deterministic_generated_at(rows)
            self.assertEqual(first, deterministic_generated_at([dict(rows[0])]))
            self.assertTrue(first.startswith("content-sha256:"))
            changed = deterministic_generated_at([{"site_id": "a", "value": 2.0}])
        self.assertNotEqual(first, changed)

    def test_source_date_epoch_is_honoured(self) -> None:
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            self.assertEqual(deterministic_generated_at([]), "1970-01-01T00:00:00Z")


@unittest.skipUnless(_HAS_DB, "NQR SQLite export absent")
class ReproducibleOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self.db = self._tmp / "nqr.sqlite"
        shutil.copy(_SHIPPED_DB, self.db)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SOURCE_DATE_EPOCH", None)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_site_flag_export_is_byte_reproducible(self) -> None:
        jsonl = self._tmp / "site_flags.jsonl"
        first = write_consistency_flags(database_path=self.db, jsonl_path=jsonl)
        first_bytes = jsonl.read_bytes()
        second = write_consistency_flags(database_path=self.db, jsonl_path=jsonl)
        self.assertEqual(first.generated_at, second.generated_at)
        self.assertEqual(first_bytes, jsonl.read_bytes())

    def test_landolt_flag_export_is_byte_reproducible(self) -> None:
        jsonl = self._tmp / "landolt_flags.jsonl"
        first = write_landolt_review_flags(database_path=self.db, jsonl_path=jsonl)
        first_bytes = jsonl.read_bytes()
        second = write_landolt_review_flags(database_path=self.db, jsonl_path=jsonl)
        self.assertEqual(first.generated_at, second.generated_at)
        self.assertTrue(first.generated_at.startswith("content-sha256:"))
        self.assertEqual(first_bytes, jsonl.read_bytes())


if __name__ == "__main__":
    unittest.main()
