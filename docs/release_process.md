# Release Process

_Last updated: 2026-06-28_

MRSpinDynamics is released as a **single citable unit**: the whole repository is
versioned together, tagged once, published as one GitHub Release, and archived on
Zenodo to mint a citable DOI. There is no per-subproject PyPI publishing — the
subprojects are interdependent and at very different maturity levels, so a single
workspace version is the model that fits.

> If independent PyPI distribution of a specific subproject (most likely
> `PythonSpinDynamics`) is wanted later, see
> [`publishing_plan.md`](publishing_plan.md). That is a possible **future**
> direction, not the current release model.

## Versioning

- One version of record for the whole workspace, stored in
  [`CITATION.cff`](../CITATION.cff) (`version:`) and the top
  [`CHANGELOG.md`](../CHANGELOG.md).
- Subproject `pyproject.toml` files carry the same version for internal
  consistency, but they are not published independently.
- A subproject's `Development Status` classifier still reflects its own maturity
  (e.g. `PythonSpinDynamics` is more mature than `QuadrupolarDFT`); the shared
  version number is not a uniform stability claim.
- Tags use a plain `v<version>` form, e.g. `v0.1.0`.

## One-time setup (maintainer)

Citation and DOI minting need accounts and cannot be scripted in this repo:

1. Sign in to [Zenodo](https://zenodo.org/) with the GitHub account that owns the
   repository.
2. Under **Zenodo -> GitHub**, toggle the `supertjhok/MRSpinDynamics` repository
   **On**. Zenodo then archives every subsequent GitHub Release and mints a DOI.
3. (Optional) Confirm the GitHub "Cite this repository" button renders from
   [`CITATION.cff`](../CITATION.cff).

Zenodo issues a *concept DOI* (always resolves to the latest version) plus a
*version DOI* per release. After the first release, add the concept DOI to
[`CITATION.cff`](../CITATION.cff) (`doi:`) and a badge to the README.

## Cutting a release

1. Confirm CI is green on `main` (PythonSpinDynamics, QuadrupolarDFT,
   NQRDatabase, and integration workflows).
2. Bump the version everywhere it appears:
   - [`CITATION.cff`](../CITATION.cff) `version:` and `date-released:`
   - [`.zenodo.json`](../.zenodo.json) (no version field; review metadata)
   - `PythonSpinDynamics/pyproject.toml`, `QuadrupolarDFT/pyproject.toml`,
     `integration/pyproject.toml`
   - the `scripts/bump_version.py` helper does this in one step (see below).
3. Move the `Unreleased` section of [`CHANGELOG.md`](../CHANGELOG.md) to the new
   version with today's date, and add a fresh `Unreleased` header.
4. Regenerate any committed generated artifacts and confirm no drift, e.g.:

   ```powershell
   cd PythonSpinDynamics
   python docs\generate_api_reference.py
   git diff --exit-code docs\python_api\api_reference.md
   ```

5. Commit the version bump and changelog.
6. Tag and push:

   ```powershell
   git tag v0.1.0
   git push origin v0.1.0
   ```

7. The [release workflow](../.github/workflows/release.yml) creates a GitHub
   Release from the tag using the changelog section as the body. (Or create the
   Release manually with `gh release create v0.1.0 --notes-file <section>`.)
8. Zenodo archives the Release and mints the DOI. Add the DOI to
   [`CITATION.cff`](../CITATION.cff) and the README badge in a follow-up commit.

## Consistency check

`scripts/check_versions.py` fails if the versions in `CITATION.cff` and the three
`pyproject.toml` files disagree. It runs in CI so a release can never ship with a
split version.
