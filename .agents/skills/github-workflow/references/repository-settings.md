# Repository Settings

Use this reference for GitHub repository settings, including Actions policy,
workflow token permissions, selected-action allowlists, required checks,
rulesets, merge policy, immutable releases, and interaction limits. Run commands
with authenticated `gh`, replace `OWNER/REPO` with the exact reviewed target,
and restrict output to the fields needed as evidence.

## Contents

- [Inventory before changing settings](#1-inventory-before-changing-settings)
- [Limit concurrent external pull requests](#2-limit-concurrent-external-pull-requests)
- [Apply the safe fallback ruleset](#3-apply-the-safe-fallback-ruleset)
- [Make a required-check context safe](#4-make-a-required-check-context-safe)
- [Restrict Actions without breaking workflows](#5-restrict-actions-without-breaking-workflows)
- [Create or complete the default ruleset](#6-create-or-complete-the-default-ruleset)
- [Verify stored settings](#7-verify-stored-settings)

## 1. Inventory before changing settings

```powershell
$repo = 'OWNER/REPO'

$repository = gh api --method GET `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo" `
  --jq '{full_name,visibility,admin:.permissions.admin,default_branch,allow_squash_merge,allow_merge_commit,allow_rebase_merge,allow_auto_merge,allow_update_branch,delete_branch_on_merge,squash_merge_commit_title,squash_merge_commit_message}' |
  ConvertFrom-Json

if ($repository.full_name -ne $repo) {
  throw "Target mismatch: expected $repo, got $($repository.full_name)"
}
if ($repository.admin -ne $true) {
  throw "Administrator access is not verified for $repo"
}

gh api "repos/$repo/immutable-releases"
gh api "repos/$repo/actions/permissions"
gh api "repos/$repo/actions/permissions/selected-actions"
gh api "repos/$repo/actions/permissions/workflow"
gh api "repos/$repo/rulesets"
```

For a public repository, also inventory the pull-request creation cap with the
versioned endpoint in section 2.

Build a per-setting evidence map from the inventory and the read-back after
every mutation. Record the endpoint, observed value, and result as `verified`,
`unverified`, `not applicable`, or an approved exception. Treat an unavailable
setting as unverified; do not infer a fork-workflow approval policy, merge
method, ruleset bypass, or Actions permission from another repository or a
related setting.

Mutate settings only when the user explicitly requests an apply. Set every
requested baseline value explicitly even if the pre-change response omitted
it. For an audit-only request, retain omissions and inaccessible values as
unverified.

The fork-contributor approval endpoint can return `404` for personal-owner
repositories. Do not report that policy as applied in that case; record it as
unverified and use the repository settings UI or a supported API when one is
available.

## 2. Limit concurrent external pull requests

For a public repository, enable **Limit open pull requests from users without
write access** and set **Maximum open pull requests per user** to `1`. This cap
affects only users without write access. Draft pull requests do not count.
Preserve the existing bypass list unless the maintainer explicitly requests a
separately reviewed change.

The REST endpoint requires repository administrator access. Use GitHub API
version `2026-03-10`, and record the requested payload as
`{"enabled":true,"max_open_pull_requests":1}` before applying it.

```powershell
if ($repository.visibility -ne 'public') {
  throw "Pull-request creation cap is not applicable to $repo"
}

gh api --method GET `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo/interaction-limits/pulls/creation-cap" `
  --jq '{enabled,max_open_pull_requests}'

gh api --method PATCH `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo/interaction-limits/pulls/creation-cap" `
  -F enabled=true -F max_open_pull_requests=1 `
  --jq '{enabled,max_open_pull_requests}'
```

Run the PATCH only when the user explicitly requests an apply and every
preflight check succeeds. Stop on a target, visibility, or authority mismatch;
do not reinterpret it as permission to select a different repository.

Immediately repeat the GET and require both stored values to match the
requested payload. For a non-public repository, record the setting as not
applicable. If the endpoint or administrator evidence is unavailable, record
the setting as unverified; do not infer it from another repository or the
PATCH response alone.

## 3. Apply the safe fallback ruleset

When no check context has been observed on a pull request, create or update the
`default` ruleset with every other baseline rule: target the default branch,
restrict deletions and force pushes, require pull requests, allow only squash
merging, and allow repository-admin bypass only on pull requests. Omit only
`required_status_checks`; record the ruleset as incomplete and do not claim
status-check enforcement.

Use the template in section 6 after removing its `required_status_checks`
object. For an existing ruleset, preserve unrelated rules and send the complete
reviewed replacement with `PUT`.

## 4. Make a required-check context safe

1. Use `github-actions-quality-check` to add or adapt a validation workflow
   that runs the intended job on `pull_request`. Add `merge_group` when the
   repository uses a merge queue.
2. Give the job a stable visible name, such as `Check` or `Test`.
3. Merge that workflow change to the default branch.
4. Open a pull request and wait for a successful run. Confirm the exact visible
   context before adding it to the ruleset:

   ```powershell
   gh pr checks <number>
   ```

Do not require a job that runs only on `push`, a release job, or a context whose
current name was not observed on a pull request. `github-actions-quality-check`
owns the workflow implementation and event compatibility;
`github-workflow` owns context selection and the repository setting mutation.

## 5. Restrict Actions without breaking workflows

Use `github-actions-quality-check` to inventory `uses:` in workflow files and
all reachable local Composite Actions or reusable workflows. Preserve local
actions and GitHub-owned actions; allow only the external action or
reusable-workflow names that the inventory finds. Keep full-SHA pinning required
for workflow execution, but allow each selected name with `@*` so updating a
reviewed pin does not require a settings change. Do not wildcard an owner or all
actions. Build the allowlist from the inventory output; do not ask a maintainer
to translate workflow references into settings by hand.

Set `allowed_actions=selected`, `sha_pinning_required=true`,
`github_owned_allowed=true`, and `verified_allowed=false` explicitly during an
apply. Do not assume an omitted pre-change SHA-pinning value was already safe.

When the inventory finds no external `uses:` reference, set selected actions
with GitHub-owned actions allowed, Marketplace verified creators disallowed,
and no `patterns_allowed[]` entries. This is a valid least-privilege result;
downloaded tools are not Action allowlist entries and still require the
separate `security-check` review.

```powershell
$actionsPolicy = @{
  enabled = $true
  allowed_actions = 'selected'
  sha_pinning_required = $true
}

$selectedActions = @{
  github_owned_allowed = $true
  verified_allowed = $false
  patterns_allowed = @(
    'EXTERNAL_OWNER/ACTION@*'
  )
}

$workflowPermissions = @{
  default_workflow_permissions = 'read'
  can_approve_pull_request_reviews = $false
}

$actionsPolicy | ConvertTo-Json -Depth 4 | gh api --method PUT `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo/actions/permissions" --input -

$selectedActions | ConvertTo-Json -Depth 4 | gh api --method PUT `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo/actions/permissions/selected-actions" --input -

$workflowPermissions | ConvertTo-Json -Depth 4 | gh api --method PUT `
  -H 'Accept: application/vnd.github+json' `
  -H 'X-GitHub-Api-Version: 2026-03-10' `
  "repos/$repo/actions/permissions/workflow" --input -
```

Replace the placeholder with one exact `owner/action@*` or
`owner/repository/path@*` entry per observed external dependency. When the
inventory finds none, use `patterns_allowed = @()` so the payload sends an
explicit empty JSON array. Read back all three endpoints after the change. If
an organization or enterprise policy prevents a repository override, retain
the inherited restriction as observed evidence and report the requested
repository value as unapplied rather than weakening the parent policy.

## 6. Create or complete the default ruleset

Save the following JSON as `ruleset.json` after replacing `Check` with an
observed pull-request check context. Repository role ID `5` is the `admin` role;
its bypass mode is limited to pull requests. If no context is available, remove
the complete `required_status_checks` object and apply the fallback from
section 3.

```json
{
  "name": "default",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [
    { "actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request" }
  ],
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "pull_request", "parameters": { "allowed_merge_methods": ["squash"] } },
    { "type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": true,
      "do_not_enforce_on_create": false,
      "required_status_checks": [{ "context": "Check" }]
    } }
  ]
}
```

```powershell
gh api --method POST "repos/$repo/rulesets" --input ruleset.json
```

For an existing ruleset, fetch its detail before changing it. Preserve every
unrelated rule; identify extra bypass actors and other exceptions separately
instead of silently treating them as the administrator baseline. Send only the
complete, reviewed replacement to `PUT repos/$repo/rulesets/<id>`.

## 7. Verify stored settings

```powershell
gh api "repos/$repo/rulesets" --jq '.[] | select(.name == "default") | .id'
gh api "repos/$repo/rulesets/<id>"
```

Confirm the target is `~DEFAULT_BRANCH`, deletion and force pushes are
restricted, pull requests allow only squash merging, the observed check is
required, and the only baseline bypass is repository-admin with `pull_request`
mode. Also read back the repository merge settings, immutable releases, both
Actions policy endpoints, selected-action policy, and workflow token
permissions; mark every unavailable value unverified. For a public repository,
read back the pull-request creation cap with API version `2026-03-10` and
confirm the requested stored values.

Authoritative sources for the pull-request creation cap:

- [GitHub repository interaction limits](https://docs.github.com/en/communities/moderating-comments-and-conversations/limiting-interactions-in-your-repository)
- [REST API endpoints for repository interactions](https://docs.github.com/en/rest/interactions/repos?apiVersion=2026-03-10)

Authoritative source for Actions permissions and the selected-action
allowlist:

- [REST API endpoints for GitHub Actions permissions](https://docs.github.com/en/rest/actions/permissions?apiVersion=2026-03-10)
