# PythonSpinDynamics Publishing Plan

_Last updated: 2026-06-28_

This plan turns roadmap item **Publish** into a repeatable release process for
`PythonSpinDynamics`: beta versioning, PyPI/TestPyPI publication, MkDocs-hosted
documentation, and a small checklist that can be run the same way for every
release.

## Goal

Move `PythonSpinDynamics` from an internal source-tree package to a public beta
release that users can install with `pip`, cite, and browse online.

The first target should be a beta release such as:

```toml
version = "0.1.0b1"
classifiers = [
  "Development Status :: 4 - Beta",
  ...
]
```

Use the beta cycle to exercise packaging, documentation, and installation
without pretending that every workflow is final. Promote to `0.1.0` only after
one or two beta/TestPyPI cycles install cleanly and the known gaps remain
clearly documented.

## Beta Readiness Criteria

Before publishing `0.1.0b1`, the package should satisfy:

- Fresh wheel install works in a clean Python environment.
- CI smoke matrix is green across supported Python and OS combinations.
- Full validation job is green on Ubuntu / Python 3.12.
- `python -m ruff check src tests examples` passes from a clean checkout.
- `python docs/generate_api_reference.py` leaves
  `docs/python_api/api_reference.md` unchanged.
- `mkdocs build --strict` succeeds.
- Public workflow APIs, examples, known gaps, and validation status are
  discoverable from the docs site.
- TestPyPI publish and install round trip succeeds before PyPI publish.

## Package Metadata Work

Update `PythonSpinDynamics/pyproject.toml`:

- Set `version = "0.1.0b1"` for the first beta.
- Change classifier from `Development Status :: 3 - Alpha` to
  `Development Status :: 4 - Beta`.
- Add `license-files` or otherwise ensure the GPL license text is included in
  source and wheel distributions.
- Add `[project.urls]`, for example:
  - `Homepage`
  - `Documentation`
  - `Source`
  - `Issues`
- Add `maintainers` if desired.
- Add optional extras:
  - `docs = ["mkdocs", "mkdocs-material", ...]`
  - `release = ["build", "twine"]`
- Keep runtime dependencies minimal: core should remain NumPy-only unless a
  dependency is required at import time.

Update ignore rules:

- Ignore `build/`, `dist/`, `site/`, and `*.egg-info/`.
- Do not track generated packaging metadata.

## Documentation Site

Add `PythonSpinDynamics/mkdocs.yml` and use the existing Markdown docs as the
first public site:

- `docs/python_api/index.md`
- `installation.md`
- `concepts.md`
- `workflows.md`
- `examples.md`
- `api_reference.md`
- feature pages for NQR, ESR, exchange, analysis, internal gradients, etc.
- selected validation and known-gaps pages.

Recommended first theme: `mkdocs-material`, because it gives good navigation,
search, and code formatting with little custom work. Keep the configuration
simple until the docs site stabilizes.

The docs workflow should:

1. Install `.[docs]`.
2. Run `python docs/generate_api_reference.py`.
3. Fail if the generated API reference differs from the committed file.
4. Run `mkdocs build --strict`.
5. Deploy to GitHub Pages from CI, not from a local dirty worktree.

MkDocs has a built-in GitHub Pages deploy path (`mkdocs gh-deploy`), but local
deploys can accidentally include untracked files. Prefer GitHub Actions so the
site is built from a clean checkout.

## PyPI/TestPyPI Publishing

Use PyPI Trusted Publishing with GitHub Actions rather than long-lived API
tokens. Trusted Publishing uses GitHub OIDC to mint short-lived credentials for
the configured project and workflow.

One-time setup:

- Create or reserve the `python-spin-dynamics` project on TestPyPI and PyPI, or
  configure pending trusted publishers for first upload.
- Configure trusted publishers for repository `supertjhok/MRSpinDynamics`.
- Use a dedicated workflow file, e.g.
  `.github/workflows/python-spin-dynamics-release.yml`.
- Configure GitHub Environments:
  - `testpypi`
  - `pypi`
- Require manual approval for the `pypi` environment.

Release workflow shape:

- Trigger on `workflow_dispatch` and tags matching
  `python-spin-dynamics-v*`.
- Build from `PythonSpinDynamics`.
- Run the same smoke gates as the normal CI.
- Build distributions with `python -m build`.
- Validate distributions with `python -m twine check dist/*`.
- Install the built wheel into a fresh environment and import representative
  modules.
- Upload `dist/*` as workflow artifacts.
- Publish to TestPyPI for beta/manual test runs.
- Publish to PyPI only for approved release tags.

Tag convention:

```text
python-spin-dynamics-v0.1.0b1
python-spin-dynamics-v0.1.0
```

This avoids ambiguity in a monorepo that may eventually publish more than one
Python distribution.

## Release Checklist

For each release:

1. Confirm `main` is green.
2. Update `PythonSpinDynamics/pyproject.toml` version.
3. Update release notes / changelog.
4. Regenerate API reference:

   ```powershell
   cd PythonSpinDynamics
   python docs\generate_api_reference.py
   git diff --exit-code docs\python_api\api_reference.md
   ```

5. Run smoke gates:

   ```powershell
   python -m unittest tests.smoke_tests
   python -m ruff check src tests examples
   ```

6. Build and check distributions:

   ```powershell
   python -m build
   python -m twine check dist/*
   ```

7. Create and push the release tag:

   ```powershell
   git tag python-spin-dynamics-v0.1.0b1
   git push origin python-spin-dynamics-v0.1.0b1
   ```

8. Let the release workflow publish to TestPyPI.
9. Verify install from TestPyPI in a fresh environment.
10. Approve the PyPI environment for the real publish.
11. Verify install from PyPI.
12. Verify the MkDocs site reflects the release.

## First Implementation PR

The first publish-process PR should be small and mechanical:

- Add `mkdocs.yml`.
- Add `docs` and `release` optional extras.
- Add release metadata and project URLs.
- Add `.github/workflows/python-spin-dynamics-docs.yml`.
- Add `.github/workflows/python-spin-dynamics-release.yml`.
- Add `PythonSpinDynamics/RELEASING.md` or link this plan from the package docs.
- Update `.gitignore` for build artifacts.
- Add a package-build check to CI.

Avoid combining this with unrelated physics or performance changes.

## Later

- Add a changelog generator or a manually curated `CHANGELOG.md`.
- Add versioned documentation with `mike` after the first stable release.
- Add Zenodo DOI integration once releases are tagged consistently.
- Prepare JOSS metadata after the docs site and PyPI package are stable.
- Consider package split or separate distributions only if the monorepo starts
  publishing multiple independently versioned packages.

## References

- PyPI Trusted Publishing:
  <https://docs.pypi.org/trusted-publishers/>
- PyPA guide for publishing with GitHub Actions:
  <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>
- Python packaging tutorial:
  <https://packaging.python.org/en/latest/tutorials/packaging-projects/>
- MkDocs deployment guide:
  <https://www.mkdocs.org/user-guide/deploying-your-docs/>
