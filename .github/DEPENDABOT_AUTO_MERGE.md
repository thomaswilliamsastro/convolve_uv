# Dependabot auto-merge

`.github/workflows/dependabot-auto-merge.yml` automatically approves
Dependabot pull requests and enables GitHub's native auto-merge for them.
It never merges anything itself and never bypasses required checks or
reviews - it only:

1. Approves the pull request (`gh pr review --approve`), and
2. Enables auto-merge (`gh pr merge --auto --squash`).

GitHub itself performs the actual merge, and only once **all** required
status checks pass and all required reviews are satisfied.

## Why `pull_request`, not `pull_request_target`

The workflow triggers on `pull_request`, never checks out the PR's code,
and never runs anything from the PR diff - it only calls the GitHub API
via `gh` using trusted event metadata (`github.event.pull_request.*`).
This avoids the classic `pull_request_target` + "checkout untrusted PR
head" privilege-escalation pattern, at the cost of nothing here, since the
job doesn't need to build or test the code (that's what `tests.yml` /
`build.yml` are for).

## Required repository settings

For this workflow to work, the following repository settings must be
enabled (Dependabot pull requests already satisfy the "same repo, not a
fork" assumption these rely on):

- **Settings → Actions → General → Workflow permissions →
  "Allow GitHub Actions to create and approve pull requests"** must be
  checked. Without this, `gh pr review --approve` fails with
  `GitHub Actions is not permitted to approve pull requests`.
- **Settings → General → Pull Requests → "Allow auto-merge"** must be
  enabled for the repository. Without this, `gh pr merge --auto` fails.
- **Branch protection on `main`** should require the status checks this
  repository already runs on pull requests (e.g. `Test`, `Build`,
  `Check Changelog`) before merging, so GitHub's auto-merge cannot
  complete until they pass. If branch protection requires **more than
  one** approving review, a single automated approval from this workflow
  will not be sufficient on its own to satisfy the rule.
- Do not enable "Require approval of the most recent push" together with
  a policy that forbids the same actor from approving; this workflow's
  approver is `github-actions[bot]`, distinct from the PR author
  (`dependabot[bot]`), so self-approval restrictions are not a concern.

No personal access token or extra secret is used or required - only the
default `GITHUB_TOKEN`, scoped narrowly to `pull-requests: write` and
`contents: write` for this one job.

## Interaction with `dependabot-changelog.yml`

`dependabot-changelog.yml` pushes a changelog commit to the same PR
branch, which fires its own `synchronize` event. If branch protection
dismisses stale reviews on push, that commit could invalidate this
workflow's earlier approval. `dependabot-auto-merge.yml` re-runs on every
`synchronize` event for the PR (including the one caused by the changelog
commit) and re-approves each time, so the PR is never left stuck waiting
on a dismissed review. Auto-merge itself, once enabled, persists across
new commits and does not need to be re-enabled.
