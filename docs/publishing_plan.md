# PythonSpinDynamics Package-Index Publishing Plan (deferred option)

_Last updated: 2026-07-11_

> **Status: artifact readiness and hosted documentation are complete; public
> index publication remains deferred.** MRSpinDynamics is released as a
> single citable workspace unit (one repo version, one GitHub Release, one Zenodo
> DOI) — see [`release_process.md`](release_process.md). This document describes a
> **possible future** direction: publishing `PythonSpinDynamics` as an
> independently versioned PyPI distribution. It is kept for reference and should
> only be revisited if there is concrete demand for `pip install`-able subpackages.
> Note the dependency ordering it would require: `python-spin-dynamics` would have
> to be on PyPI before `quadrupolar-dft`, and both before `mr-integration` (which
> imports them), so a standalone PyPI release of the whole stack is non-trivial.

PythonSpinDynamics now builds and clean-installs both a wheel and source
distribution in CI, exposes the `spin-dynamics` console command, ships
`py.typed`, checks metadata with Twine, gates MyPy, and enforces branch
coverage. This document therefore starts at the remaining boundary:
TestPyPI/PyPI publication, hosted documentation, and release operations.

## Goal

Publish the already-buildable package to a public index so users can install it
without a checkout and browse version-matched hosted documentation.

The package is currently versioned as 0.2.0 and remains classified Alpha. A
future index publication should use the next workspace-approved pre-release or
release version, for example:

```toml
version = "0.3.0b1"
classifiers = [
  "Development Status :: 4 - Beta",
  ...
]
```

Use the beta cycle to exercise publication and hosted documentation without
pretending that every workflow is final. Promote only after one or two
beta/TestPyPI cycles install cleanly and the known gaps remain clearly
documented.

## Publication Readiness Criteria

Before the first TestPyPI upload, confirm:

- [x] Fresh wheel and sdist installs work in clean Python environments.
- [x] CI smoke matrix covers supported Python and OS combinations.
- [x] Installed console entry point and `py.typed` marker are verified.
- [x] Static typing and a 70% branch-coverage floor are gated.
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

- Set the agreed next beta version (for example `0.3.0b1`).
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

Implemented independently of public package-index publication. The searchable
site is configured by `PythonSpinDynamics/mkdocs.yml`, built strictly in CI, and
deployed from clean `main` commits to
<https://supertjhok.github.io/MRSpinDynamics/>. Existing Markdown is the web
manual, the generated API and validation matrix are checked before deployment,
and `docs/user_manual.pdf` remains the downloadable print edition.

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
python-spin-dynamics-v0.3.0b1
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
   git tag python-spin-dynamics-v0.3.0b1
   git push origin python-spin-dynamics-v0.3.0b1
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
