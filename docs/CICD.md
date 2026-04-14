# CI/CD Pipeline

This document describes the CI/CD pipeline for this repository. All workflows are defined under [`.github/workflows/`](../.github/workflows/).

Assuming the current base version is `1.0.0` and the previous dev build was `1.0.0-dev1`.

---

## Branch Model

```
feature/issue-001/name    →    develop    →    main    →    GitHub Release
```

- **feature branches** — day-to-day development, merged into `develop` via PR
- **develop** — integration branch; triggers the edge Docker build
- **main** — staging/acceptance; triggers the release-candidate promotion
- **GitHub Release** — production promotion; triggered manually from the GitHub UI

---

## Release Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  feature/issue-001/name-of-your-issue                               │
|         (feature branch)                                            │
└────────────────────┬────────────────────────────────────────────────┘
                     │ PR merged to develop
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│           publish-edge.yml                                          │
|  · Build Docker image from source                                   │
│  · Tag Docker image                                                 │
│      :edge                                                          │
│      :v1.0.0-dev1                                                   │
│  · Create git tags                                                  │
│      1.0.0-dev1 (no v — used by versioning)                         │
│      edge                                                           │
└────────────────────┬────────────────────────────────────────────────┘
                     │ PR merged to main
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│           pre-release.yml                                           |
|  · Retag (no rebuild)                                               │
│      :edge → :release-candidate                                     │
│  · Create GitHub pre-release                                        │
│  · Build and publish Python package                                 │
│      swissgeol_boreholes_dataextraction-1.0.0-py3-none-any.whl      │
│  · Create git tag                                                   │
│      1.0.0 (no v — signals full release to versioning)              │
└────────────────────┬────────────────────────────────────────────────┘
                     │ GitHub Release published (manual)
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│             release.yml                                             │
│  · Retag (no rebuild)                                               │
│      :release-candidate →                                           │
│          :v1.0.5                                                    │
│          :v1.0                                                      │
│          :v1                                                        │
│          :latest                                                    │
│  · Build and publish Python package                                 │
│      swissgeol_boreholes_dataextraction-1.0.5-py3-none-any.whl      │
│  · Open PR: mark v1.0.5 as released in CHANGELOG.md                 |
└─────────────────────────────────────────────────────────────────────┘
```

Docker images are **built once** (on `develop`) and **promoted by retagging** — the same binary progresses through all stages without being rebuilt.

---

## Quality Checks

These run automatically on every pull request to `develop` or `main`.

| Workflow | Trigger | What it does |
|---|---|---|
| `pytest.yml` | pull request to `develop`/`main` | Runs the test suite with coverage; posts a coverage comment on PRs |
| `pre-commit.yml` | pull request to `develop`/`main` | Runs all pre-commit hooks (linting, formatting) |
| `pipeline_run.yml` | pull request to `develop`/`main` | Runs an end-to-end pipeline example to verify nothing is broken |

---

## Manual Workflows

| Workflow | Purpose |
|---|---|
| `publish-edge-package.yml` | Builds and publishes a Python wheel from the current `:edge` image. Use this to test an unreleased build in another application before it is promoted to `release-candidate`. Accepts an optional `base` input to target a specific dev tag (e.g. `v1.0.0-dev1`). |
| `pre-release.yml` (dispatch) | Re-runs the release-candidate promotion manually. Useful if the automatic push-to-main trigger needs to be re-run. Accepts an optional `base` input to target a specific image tag. |
| `release.yml` (dispatch) | Re-runs the stable release promotion manually. Requires a `TAG_NAME` matching the version embedded in the `:release-candidate` image. |

---

## Docker Tag Reference

| Tag | Source | Environment |
|---|---|---|
| `:edge` | Latest `develop` build | DEV |
| `:v1.0.0-dev1` | Specific dev build | DEV |
| `:release-candidate` | Latest `main` promotion | INT / Staging |
| `:latest` | Latest stable release | PROD |
| `:v1` | Latest release in major version 1 | PROD |
| `:v1.0` | Latest release in minor version 1.0 | PROD |
| `:v1.0.5` | Exact patch release | PROD |

---

## Python Package Reference

Python packages are published as assets on GitHub Releases and can be installed directly:

```bash
# Edge build (for testing unreleased changes)
pip install https://github.com/swisstopo/swissgeol-boreholes-dataextraction/releases/download/v1.0.0-dev99/swissgeol_boreholes_dataextraction-1.0.0-dev99-py3-none-any.whl

# Stable release
pip install https://github.com/swisstopo/swissgeol-boreholes-dataextraction/releases/download/v1.0.5/swissgeol_boreholes_dataextraction-1.0.5-py3-none-any.whl
```