---
name: apm-workflow
description: Select a reviewed APM CLI version, then set up, pin, deploy, audit, and update APM-managed agent dependencies safely. Use when creating or editing apm.yml or apm.lock.yaml, choosing or installing APM, adding an Agent Skill, plugin, or MCP dependency, validating a pinned deployment, or preparing a cooldown-aware update proposal.
---

# APM Workflow

Keep agent context reproducible and reviewable. Select the newest reviewed APM
release that has completed the seven-day cooldown or has a manifest-recorded
maintainer exception for a specific unmet gate. Filesystem location and local
recency do not establish eligibility. Require full commit pins, lockfile
hashes, and review for every changed third-party dependency.

## 1. Inspect before changing

1. Locate `apm.yml`, `apm.lock.yaml`, `apm-policy.yml`, existing agent
   directories, and repository guidance. Read `apm_version` and
   `lockfile_version` from an existing lockfile before running a mutating APM
   command; treat them as compatibility evidence, not an instruction to keep an
   older CLI forever.
2. Read [the bootstrap manifest](references/apm-bootstrap.json). Its version is
   the newest APM release whose provenance, artifacts, checksums, publication
   date, and cooldown status this Skill has reviewed. When
   `cooldown_exception` is present, verify that it identifies the specific
   unmet gate, reason, approval time, and scope. Do not infer a broader waiver.
   Do not prefer a newer local `current`, `PATH`, project-local, or legacy
   executable merely because it is easier to find.
3. Keep an unpublished consumer project at `version: 0.0.0` in `apm.yml`.
   Change that version only after its distribution and versioning design is
   explicitly decided.
4. Inventory `apm` on `PATH`, project-local copies, and locations reported by
   the current official installation documentation and the platform's command
   resolver. Record each resolved executable and `--version`; path precedence
   does not override cooldown eligibility.

   Select the manifest version when it is already installed and invoke an
   off-PATH copy by absolute path. Do not modify PATH to satisfy one repository
   task. If the selected version is absent, use the verified temporary-artifact
   flow in section 2 instead of installing it implicitly.
5. If the lockfile records an older APM version than the selected eligible
   version, capture it for comparison and regenerate it from scratch with the
   selected executable. Do not edit `apm_version` by hand: remove only the
   validated project lockfile, run `apm lock`, review the complete result, and
   then run the frozen install and audit checks. Treat substantial lock-format
   churn separately from a dependency update.
6. If the lockfile records a newer, not-yet-eligible APM version, do not execute
   that CLI merely to preserve its generator version. Report the mismatch and
   wait for maintainer direction before the destructive regeneration in step 5.
   After that direction, capture and regenerate the lock by the same procedure
   as step 5. Selecting the newest normally eligible CLI is not a cooldown
   exception.
7. Use the official lockfile migration guidance to understand schema changes,
   but do not substitute `apm install` for the from-scratch regeneration in
   step 5. Restore and stop if a frozen or allegedly read-only command rewrites
   the lockfile unexpectedly.
8. If `apm.yml` exists, preserve it. Do not run `apm init --yes`, because it
   overwrites an existing manifest.
9. If no manifest exists, inspect the repository's agent targets and run
   interactive `apm init` from the project root. Select only the targets the
   project actually uses; review the generated `apm.yml` before adding deps.
10. Treat the CLI and agent dependencies as supply-chain-sensitive code. Check the
   repository policy and any organization policy before proceeding.

### 1.1 Regenerate a lockfile without losing deployment ownership

When a CLI migration or ledger repair requires deleting and rebuilding the lockfile,
preserve the old lockfile's ownership evidence before removing it. Generated target
files can remain on disk after the ledger is gone; a new resolver may then classify
them as unmanaged local content, skip the collision, and silently omit a requested
Skill or resource from the new lockfile.

1. From the old lockfile, record the selected packages and Skill subsets, deployment
   owners, deployed paths, content hashes, target roots, and generator version. Compare
   this snapshot with `apm.yml` and the filesystem before changing anything.
2. Identify collisions in the actual target root. Distinguish paths owned by the old
   APM ledger from project-authored Skills, files owned by another package, and unknown
   local content. Never move or delete a path whose ownership is ambiguous.
3. Before removing the old lockfile, move only the exact, verified APM-owned generated
   collision paths to a task-specific temporary backup outside the repository. Preserve
   relative paths and hashes so the old deployment can be restored. Do not use
   `--force` to overwrite a collision or clear the entire agent target.
4. Remove only the validated project lockfile, regenerate it with the selected CLI, and
   run the frozen install. A skipped-file or collision warning is a blocker, even when
   the command exits successfully.
5. Compare the new package list, requested Skill subsets, deployed paths, and target
   roots with the manifest and the old snapshot. Every removed or added path needs an
   expected dependency or generator explanation; an unexpectedly missing selected
   Skill is lock-generation failure.
6. Run `apm audit --ci` and directly verify representative restored files and hashes.
   If regeneration, installation, comparison, or audit fails, restore the old lockfile
   and backed-up paths instead of accepting a partial deployment.
7. Delete the temporary backup only after the new lockfile, deployed output, diff, and
   audit all pass. Record any ownership ambiguity or tool collision behavior in the
   handoff.

## 2. Obtain the selected eligible APM version safely

Reuse an installed copy only when its exact version matches the selected
reviewed release. Do not run an unpinned installer or `apm self-update` merely
to obtain whatever version is newest upstream.

Treat installation, self-update, PATH changes, and replacement of the active
APM binary as user-environment changes. A request to maintain one repository
does not authorize them.

Read [the bootstrap manifest](references/apm-bootstrap.json), then consult the
current [official installation guide](https://microsoft.github.io/apm/getting-started/installation/),
[`apm self-update` reference](https://microsoft.github.io/apm/reference/cli/self-update/),
and installer source in the
[official APM repository](https://github.com/microsoft/apm). Do not reproduce
their commands, paths, environment variables, or installer behavior in this
Skill: those are upstream-owned and may change. The manifest remains this
Skill's source of truth for the reviewed version, release date, artifacts, and
SHA-256 values.

When the selected version is not installed:

1. Download the exact official release archive named in the bootstrap manifest
   to a unique operating-system temporary directory. Verify its hard-coded
   SHA-256 before extraction, extract the complete bundle, and invoke the
   packaged executable by absolute path. Do not add it to PATH, copy a shim,
   place it in the repository, or alter the active APM installation.
2. Confirm `--version` exactly matches the manifest. If the official artifact
   is not portable on the current platform, cannot preserve its adjacent
   runtime files, or fails integrity verification, stop instead of inventing an
   installation layout or choosing an ineligible version.
3. Remove only the validated task-specific temporary directory after the work
   finishes. Record the archive source, expected and observed SHA-256,
   executable path, and version in the handoff.
4. If the user explicitly requests a persistent installation, then consult the
   current official sources above and use their documented version-pin and
   integrity mechanisms. Let the official installer own its layout. Do not
   alter PATH or replace an active binary without explicit authorization for
   those specific environment changes.
5. For an enterprise mirror, use HTTPS, set `APM_NO_DIRECT_FALLBACK=1`, and
   verify that the mirror serves the identical reviewed artifact and checksum.
   Do not treat this as a substitute for review of the artifact.

## 2.1 Propose a bootstrap-version refresh

Do not change the bootstrap manifest automatically. Use this proposal flow:

1. Record the current and candidate tags, release URLs, publication dates,
   platform assets, SHA-256 values, and the candidate cooldown date. Start with
   the newest candidate that has completed the cooldown, not merely the newest
   published or locally installed version. A maintainer may instead select one
   exact newer release and approve a cooldown exception for a concrete defect
   or operational blocker.
2. Treat cooldown as a minimum gate, not adoption approval. Review provenance,
   release notes, installer and artifact integrity, and compatibility from the
   current official sources. Require explicit maintainer approval.
3. Run
   `uv run --no-project --no-config --locked --script <apm-workflow-skill>/scripts/propose_bootstrap_update.py`
   to isolate the locked helper from the consumer repository's uv configuration
   and collect an eligible candidate without changing files. Attach its JSON
   output to the proposal.
4. After approval, obtain the candidate through section 2's temporary
   artifact flow and run representative lock, frozen-install, audit, and
   deployed-output checks without changing the active installation or PATH.
   A reproducible resolver, lock migration, install, audit, or output failure
   rejects the candidate even after cooldown; keep the current bootstrap and
   record the evidence instead of trying a newer unreviewed release.
5. If the candidate passes, run the same isolated command with
   `--write --confirm-version <candidate>`. It updates only
   `references/apm-bootstrap.json` and refuses an unconfirmed or mismatched
   version. Review the manifest diff before committing.

For a maintainer-approved cooldown exception, pass the exact
`--candidate-version`, `--cooldown-exception-reason`, and deterministic
`--now` approval timestamp first without `--write`, then repeat with
`--write --confirm-version <candidate>`. The helper records
`cooldown_exception` in the manifest and refuses an exception without an exact
candidate and reason. The exception waives only the time gate for that
bootstrap CLI release; provenance, checksums, artifact inspection,
compatibility tests, and dependency review remain required.

Do not use `apm self-update` to perform this refresh.

## Reference sources

| Source | Use |
| --- | --- |
| [Official APM repository](https://github.com/microsoft/apm) | Verify supported configuration, CLI behavior, and upstream guidance. |
| [APM documentation](https://microsoft.github.io/apm/) | Consult the maintained usage and reference documentation. |
| [Official installation guide](https://microsoft.github.io/apm/getting-started/installation/) | Use the supported version pin and installer-managed locations. |
| [Official migration guide](https://microsoft.github.io/apm/troubleshooting/migration/) | Handle lockfile schema and CLI upgrades without inventing a migration path. |
| [`apm self-update` reference](https://microsoft.github.io/apm/reference/cli/self-update/) | Distinguish CLI updates from dependency updates and confirm official rollback behavior. |
| [APM releases](https://github.com/microsoft/apm/releases) | Review release tags, publication dates, assets, and release notes. |
| [GitHub Releases API](https://api.github.com/repos/microsoft/apm/releases?per_page=30) | Obtain machine-readable release metadata; this is the endpoint used by the proposal script. |
| [APM security policy](https://github.com/microsoft/apm/security/policy) | Follow the upstream vulnerability-reporting process. |
| [Pinned bootstrap manifest](references/apm-bootstrap.json) | Obtain this skill's reviewed version, eligible date, asset names, and SHA-256 values. |

## 3. Add an Agent Skill or dependency

Use this staged workflow for every remote skill, prompt, agent, plugin, hook,
or MCP dependency:

1. Identify the exact source repository, optional virtual subdirectory,
   release/tag, and full 40-character commit SHA. Review the source, its
   release history, declared capabilities, transitive dependencies, and changes
   that could execute code or add MCP access.
2. Verify from an authoritative source that the adopted content is at least
   seven days old. For a virtual subdirectory in a monorepo, identify the last
   commit that changed that subdirectory at the selected ref and calculate the
   cooldown from that commit, not from an unrelated later commit elsewhere in
   the repository. If its age, provenance, or required hash cannot be verified,
   stop and report the blocker. Do not bypass the cooldown without a documented
   maintainer exception.
3. Before committing a third-party skill, determine its license from an
   authoritative upstream license file or policy. Add or update the
   repository-root `THIRD_PARTY_NOTICES.md` with its source repository and
   virtual path, full commit SHA, license identifier and canonical text or URL,
   plus any supplied copyright or NOTICE text. Do not put this record in an
   APM-managed target directory such as `.agents/skills/`. Treat a missing or
   unclear license or required notice as a blocker unless a maintainer records
   an explicit exception.
4. Add the dependency to `apm.yml` with the full commit SHA, never a default
   branch, `latest`, or an unbounded range. When selecting multiple Skills from
   one repository at one ref, use one object-form dependency with `git`, `ref`,
   and `skills`; do not encode each Skill as a separate virtual dependency.
   Keep separate entries only when their repository, ref, or dependency policy
   differs. Keep MCP entries under the same review gate and do not put tokens
   in tracked YAML.
5. Run `apm lock` to resolve and download without deploying to agent targets.
   Review `apm.lock.yaml`: each dependency must resolve to the expected commit
   and carry its content hash. In PowerShell checks, wrap variable-cardinality
   query results in `@(...)` before using `.Count`, indexing, or membership
   tests; one result otherwise becomes a scalar while multiple results become
   an array. Assert the expected count and element type. Commit the manifest
   and lockfile together.
6. Run `apm install --frozen` only after that review. It must reproduce the
   reviewed lockfile and must not resolve a new dependency. Never use
   `--force`, `--allow-insecure`, or an insecure HTTP source unless an explicit
   documented exception authorizes it.
7. Run `apm audit --ci` after deployment. Fix lockfile, integrity, drift, or
   policy findings rather than suppressing them. Add `apm install --frozen` and
   `apm audit --ci` to the project's existing CI when the user requests CI
   enforcement.

### 3.1 Maintain a packaged Skill collection

Apply this section only when a repository publishes Skill copies in both an
authoring target such as `.agents/skills/` and a package directory such as
`.apm/skills/`.

1. Identify the canonical source and each distributed copy. Use a supported
   generator when one exists; otherwise update every copy in the same change.
2. Keep validation artifacts out of canonical package trees before locking or
   deployment. Run import-based validators with bytecode writes disabled, such as
   with `PYTHONDONTWRITEBYTECODE=1`, or against a disposable copy. Inventory ignored
   as well as tracked files for transient caches, and remove only exact task-created
   artifacts. If a transient file enters a candidate ledger, discard that candidate
   and regenerate from a clean canonical tree. Restore the prior reviewed lockfile
   first when one exists; for a first lock, remove only the exact uncommitted
   task-created candidate. Do not patch, normalize, or publish a contaminated ledger.
3. Compare relative file sets and content hashes before release. A missing,
   extra, or different file is a release blocker, not a cosmetic discrepancy.
4. After `apm install --frozen`, record the installer-reported target and
   verify the named Skill at that actual target, rather than assuming a
   configured or conventional directory was used.
5. If `apm audit --ci` resolves a different target root, do not call the audit
   a pass or failure without direct evidence. Record the discrepancy, verify
   the installed file and lockfile hashes directly, and report the observed
   behavior for follow-up.
6. After deployment, review the staged diff as well as the working tree. An
   installer can rewrite unchanged text with a different line ending; stage
   only the expected manifest, lockfile, canonical, and deployed outputs, then
   use `git diff --cached --check` and a content review to distinguish a real
   generated-file delta from line-ending noise.

### 3.2 Rename or remove packaged local Skills

Apply this section when a local packaged Skill is renamed, consolidated, or
removed. Treat it as a deployment-ledger change, not a documentation-only
rename.

1. Before editing, inventory the retiring and destination source folders,
   deployed paths, resource moves, README index rows, repository references,
   and every matching entry in `apm.lock.yaml` (`deployments`, deployed-file
   lists, and local deployed-file hashes).
2. Move or remove canonical content and update callers first. Preserve
   deterministic scripts and directly linked references when their behavior
   remains supported; do not recreate them gratuitously. Regenerate every
   managed deployment from the canonical source rather than hand-editing
   `.agents/` output.
3. Regenerate the lockfile from scratch with the selected reviewed APM CLI
   whenever the existing ledger still names a retired deployment path, or when
   a normal lock/install cycle cannot prove that removed paths left the ledger.
   Do not patch lockfile hashes or deployment entries by hand. Review that no
   retired path remains before running `apm install --frozen`.
4. Run `apm install --frozen` and `apm audit --ci`. Treat missing expected
   files, orphaned retired files, unintegrated outputs, or a source/deployed
   file-set mismatch as release blockers. If the available generator cannot
   remove a retired generated path, stop and record the tool limitation rather
   than claiming the deployment is clean.
5. Review the complete canonical, deployment, and lockfile diff together.
   Confirm the installed target contains the destination Skill exactly once and
   no retired Skill name, resource, or hash record remains.

## 4. Propose updates; do not silently apply them

Pinned dependencies stay unchanged until a maintainer approves an update.
Create an update proposal, issue, or pull request with no deployment changes
first:

1. Run `apm outdated` and `apm update --dry-run` to discover candidates
   without writing files.
2. For each candidate, record the current and proposed tag and full commit
   SHA, upstream release URL and date, SHA-256/content-hash changes, source
   review summary, new transitive dependencies, and any MCP, hook, permission,
   or target-output changes.
3. State the earliest eligible adoption date: release date plus seven full
   days. Keep the proposal open until that date and explicit maintainer
   approval; a newer upstream release restarts its own cooldown.
4. After approval, run interactive `apm update` for only the approved package.
   Review the plan before accepting it, then review the `apm.yml`,
   `apm.lock.yaml`, and deployed-file diff. Run `apm audit --ci` and commit the
   result. Re-check the upstream license and notices; update the matching
   `THIRD_PARTY_NOTICES.md` entry in the same commit when the source, virtual
   path, commit, license, or attribution changes.
5. In CI and ordinary repeat installs, keep using `apm install --frozen`.
   Update discovery and update application must remain separate actions.

## 5. Reconcile third-party notices

Before handoff, reconcile `THIRD_PARTY_NOTICES.md` with the reviewed lockfile:

1. For every third-party skill whose files are committed from an APM target,
   keep one current notice entry with the source, selected virtual path, locked
   SHA, license, and required attribution.
2. When removing a dependency, run `apm prune` and confirm its deployed files
   are gone before removing its notice entry. Retain the notice when a copy or
   derivative remains in the repository.
3. Treat a changed license, copyright, or NOTICE file as an adoption-review
   change, not a metadata-only update. Resolve it before deploying the new
   version.

## Completion checklist

- `apm.yml` is present and was not overwritten accidentally.
- An unpublished project keeps `version: 0.0.0` until its distribution and
  versioning design is approved.
- The selected APM CLI was the newest reviewed release that completed the
  seven-day cooldown or had a manifest-recorded maintainer exception for the
  specific unmet gate; its absolute path and version were recorded.
- No PATH, active APM installation, or other user-environment state changed
  without explicit authorization.
- Any lock-format migration was explicit and reviewed separately from resolved
  dependency changes; unexpected lockfile writes were restored and reported.
- Every remote dependency is pinned to a reviewed full commit SHA.
- `apm.lock.yaml` is committed and contains the expected resolved commits and
  content hashes, and its `apm_version` matches the selected eligible CLI.
- Each adopted release, or the last change to a selected virtual subdirectory,
  met the seven-day cooldown or has a documented maintainer-approved exception.
- Each committed APM-deployed third-party skill has one current
  `THIRD_PARTY_NOTICES.md` entry, including its license and required notices;
  removed skills have no stale entry unless a copy remains.
- Packaged collections have matching canonical and distributed file sets and
  content hashes; the named installed Skill was checked at the
  installer-reported target.
- Deployment used `apm install --frozen`; `apm audit --ci` passed in the
  verified target context, or any target-resolution discrepancy is documented
  with direct installation and lockfile evidence.
- Updates have a reviewable, cooldown-aware proposal before application.
