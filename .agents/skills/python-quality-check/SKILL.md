---
name: python-quality-check
description: >-
  Create, align, or review strict uv-managed installable Python projects hosted
  on GitHub across packaging, dependency locking, Ruff, mypy, pytest, statement
  and branch coverage, distribution artifacts, and CI parity. Use for Python
  applications, libraries, or CLIs with pyproject.toml, .python-version,
  uv.lock, src/tests layouts, dependency changes, quality-gate adoption,
  packaging, or GitHub Actions validation; pair with a framework-specific Skill
  when one applies.
---

# Python Quality Check

## When to Use

- Use for installable Python applications, libraries, and CLIs managed with uv
  in GitHub-hosted repositories.
- Use when establishing or reviewing a repository-wide Python quality baseline.
- Pair with a framework-specific Skill for framework architecture, lifecycle,
  UI, protocol, or target-packaging requirements. The stricter applicable
  requirement wins.

## Goals

- Reproduce the complete Python environment from committed metadata and lock state.
- Keep first-party APIs, types, tests, coverage, and package boundaries explicit.
- Make local, CI, distribution, and release evidence describe the same reviewed source.
- Treat dependency, automation, build, and artifact changes as supply-chain-sensitive.

## Responsibility Boundaries

Use `code-quality-check` for general readability and maintainability,
`security-check` for dependency provenance, release age, build hooks, secrets,
and external executables, and `github-actions-quality-check` for workflow
triggers, workflow-level `permissions:` declarations, runners, and action pins.
Use `github-workflow` for repository settings and enforcement. Use
`test-quality-check` for language-independent test design, classification,
coverage policy, and overengineering audits.

This Skill owns the Python-specific baseline. Framework Skills should add only
their framework contracts and should not redefine weaker uv, Ruff, typing,
testing, coverage, or distribution rules.

## Non-Negotiable Baseline

- Use PEP 621 metadata, a real build backend, and a `src/` package layout for an
  installable project. Do not use `pythonpath = ["src"]` to disguise a package
  that cannot be installed.
- Commit `pyproject.toml`, `.python-version`, and `uv.lock`. For a new
  application, select the newest stable Python minor supported by every direct
  runtime and build dependency; require a documented compatibility reason for
  an older minor. Pin one development minor unless a documented library matrix
  requires several.
- Set `[tool.uv] exclude-newer = "P7D"` and review every dependency/lock delta
  with `security-check`.
- Put Ruff, mypy, pytest, and pytest-cov in a development dependency group.
- Run Ruff lint and format checks, strict mypy with unreachable-code warnings,
  and deterministic pytest through the locked uv environment.
- Require 100% maintained first-party statement and branch coverage. Do not
  manufacture compliance with broad omissions, import guards, or execution-only tests.
- Require keyword arguments from the first project-owned parameter at
  definitions and call sites. Keep positional parameters only for an evidenced
  external signature and document the exact exception.
- Keep local and CI commands equivalent. A check that rewrites `uv.lock` is not
  valid verification.

## Workflow

1. Inventory the Python contract.
   - Read repository guidance, `pyproject.toml`, `.python-version`, `uv.lock`,
     source, tests, scripts, workflows, build metadata, and release configuration.
   - Record project type, supported Python range, distribution/import/entry-point
     identities, build backend, dependency groups, and artifact targets.
   - Mark unavailable facts `blocked`; do not copy values from another project.
   - Run
     `uv run --no-config --locked --script <skill-root>/scripts/check_project.py <repository-root>`
     for the deterministic mechanical floor. Review every finding semantically;
     a pass does not establish behavior, artifact correctness, or release readiness.
2. Align project and dependency metadata.
   - Read [tooling-and-testing.md](references/tooling-and-testing.md) before
     editing Python metadata, Ruff, mypy, pytest, coverage, or test layout.
   - Keep runtime dependencies direct and development tools in a development
     group. Verify the newest stable compatible Python minor and newest
     cooldown-eligible uv release instead of copying an older peer value.
     Update only intended packages, inspect the full lock delta, then replay
     with `uv lock --check` and `uv sync --locked --all-groups`.
3. Align source and API quality.
   - Keep entry points thin, package imports explicit, and external effects
     behind typed boundaries.
   - Apply the keyword-only policy to project-owned functions, methods,
     dataclasses, factories, and call sites. Keep exceptions narrow and tied to
     the exact external ABI.
   - Fix lint and typing findings in code before adding a suppression. Every
     suppression must name the exact rule/code and a durable local reason.
4. Align tests and coverage.
   - Apply `test-quality-check` for behavioral scope, deterministic oracles,
     suite auditing, and the shared 100% statement/branch policy.
   - Configure pytest and Coverage.py for the selected Python source roots and
     run them through the locked uv environment locally and in CI.
5. Align CI and distribution.
   - Read [ci-and-distribution.md](references/ci-and-distribution.md) before
     changing workflows, build metadata, wheels, sdists, executables, or releases.
   - Use this Skill's `assets/github/actions/setup-python-locked` as the single
     shared implementation for committed Python/uv resolution, cache setup,
     lock verification, and synchronization. Domain Skills may compose the
     installed local action but must not carry another editable copy.
   - Re-run the complete locked validation on pull requests and on the exact
     protected integration-branch commit.
   - Build applicable distributions from a clean reviewed commit. Inspect
     archive paths, metadata, entry points, contents, permissions, licenses,
     and absence of secrets, caches, tests, and unrelated development files.
6. Verify and report.
   - Run the repository-documented commands, using this default floor when no
     stronger local command exists:

     ```shell
     uv lock --check
     uv sync --locked --all-groups
     uv run --locked ruff check .
     uv run --locked ruff format --check .
     uv run --locked mypy src tests
     uv run --locked pytest
     ```

   - Add `uv build`, safe archive inspection with
     `scripts/inspect_distribution.py`, and clean-environment install/import
     checks for installable distributions.
   - Report exact commands/results, dependency review scope, coverage statements
     and branches, artifact evidence, blockers, and every skipped check.

## Completion Checklist

- Python version, package identity, entry points, build backend, and supported
  distribution targets are explicit and consistent.
- uv lock verification and exact sync reproduce the reviewed dependency graph.
- Ruff lint/format, strict mypy, pytest, and 100% statement/branch coverage pass.
- First-party APIs are keyword-only except for documented external signatures.
- Tests establish behavior and failure/cleanup contracts, not line execution alone.
- CI is lock-preserving and equivalent to documented local checks.
- Built distributions are inspected as artifacts rather than inferred from source tests.
- Dependency, automation, and build changes have security/provenance review evidence.
