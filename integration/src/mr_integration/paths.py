"""Locate the sibling-project data files the integration layer reads.

``mr_integration`` does not bundle any data. It reads three inputs that live in
sibling subprojects of an MRSpinDynamics checkout:

- the ``NQRDatabase`` SQLite export (``NQRDatabase/data/exports/nqr.sqlite``);
- the curated DFT summary (``QuadrupolarDFT/results/nano2_efg_summary.csv``);
- the DFT structure tree (``QuadrupolarDFT/structures``).

In an editable/source checkout these are found relative to this file. An
installed wheel has no checkout next to it, so point the package at the data
with environment variables (an explicit ``database_path=`` / ``path=`` argument
always wins):

``MR_INTEGRATION_REPO_ROOT``
    Root of an MRSpinDynamics checkout; all three defaults resolve under it.
``MR_INTEGRATION_NQR_DB``
    The ``nqr.sqlite`` file itself (overrides the repository-root default).
``MR_INTEGRATION_DFT_SUMMARY``
    The DFT summary CSV (overrides the repository-root default).
``MR_INTEGRATION_STRUCTURES``
    The structure directory (overrides the repository-root default).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT_ENV = "MR_INTEGRATION_REPO_ROOT"
NQR_DB_ENV = "MR_INTEGRATION_NQR_DB"
DFT_SUMMARY_ENV = "MR_INTEGRATION_DFT_SUMMARY"
STRUCTURES_ENV = "MR_INTEGRATION_STRUCTURES"

# ``integration/src/mr_integration/paths.py`` -> repository root in a checkout.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]

_NQR_DB_RELATIVE = Path("NQRDatabase") / "data" / "exports" / "nqr.sqlite"
_DFT_SUMMARY_RELATIVE = Path("QuadrupolarDFT") / "results" / "nano2_efg_summary.csv"
_STRUCTURES_RELATIVE = Path("QuadrupolarDFT") / "structures"


def repository_root() -> Path:
    """Return the MRSpinDynamics checkout root the defaults resolve against.

    ``MR_INTEGRATION_REPO_ROOT`` takes precedence; otherwise the checkout that
    contains this source file is assumed (which is only meaningful for an
    editable/source install).
    """

    override = os.environ.get(REPO_ROOT_ENV)
    return Path(override).expanduser() if override else _CHECKOUT_ROOT


def _resolve(env_name: str, relative: Path) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser()
    return repository_root() / relative


def default_nqr_database_path() -> Path:
    """Return the default ``nqr.sqlite`` path (may not exist)."""

    return _resolve(NQR_DB_ENV, _NQR_DB_RELATIVE)


def default_dft_summary_path() -> Path:
    """Return the default DFT summary CSV path (may not exist)."""

    return _resolve(DFT_SUMMARY_ENV, _DFT_SUMMARY_RELATIVE)


def default_structures_path() -> Path:
    """Return the default DFT structure directory (may not exist)."""

    return _resolve(STRUCTURES_ENV, _STRUCTURES_RELATIVE)


def missing_data_message(kind: str, path: Path, env_name: str) -> str:
    """Return an actionable error message for a missing sibling-project input."""

    return (
        f"{kind} not found at {path}. mr_integration does not bundle data; it "
        "reads it from an MRSpinDynamics checkout. Pass the path explicitly, set "
        f"{env_name} to the file/directory, or set {REPO_ROOT_ENV} to the "
        "checkout root."
    )
