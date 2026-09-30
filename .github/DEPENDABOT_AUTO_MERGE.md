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

This workflow uses no personal access token or extra secret - only the
default `GITHUB_TOKEN`, scoped narrowly to `pull-requests: write` and
`contents: write` for this one job. (`dependabot-changelog.yml` does use a
token, see below.)

## Interaction with `dependabot-changelog.yml`

`dependabot-changelog.yml` pushes a changelog commit to the same PR
branch. The `main` ruleset requires the `Build`, `Test` and `Check
Changelog` checks to pass on the PR's latest commit (and the branch to be
up to date), so that commit needs a full CI run before auto-merge can
complete.

### Why it needs a token

A commit pushed with the default `GITHUB_TOKEN` is made by
`github-actions[bot]`. The `pull_request` runs it triggers are created, but
GitHub holds every one of them at `action_required` until a maintainer
clicks "Approve and run workflows". Until then the required checks are not
green and auto-merge cannot finish, so every Dependabot PR needed a manual
approval. (On PR #30, all six workflows for the changelog commit waited for
exactly that.)

`dependabot-changelog.yml` therefore pushes with the token in the
`DEPENDABOT_CHANGELOG_TOKEN` secret, which belongs to a real user, so the
resulting runs should start immediately (this has not yet been seen on a real
Dependabot PR; see the check below). If the secret is missing the workflow
falls back to `GITHUB_TOKEN` and prints a warning, which restores the old
behaviour (the approval click is needed).

### Setting it up

1. Create a **fine-grained personal access token** (Settings > Developer
   settings) with *Only select repositories* set to this repository, the
   single permission **Contents: Read and write** (Metadata: Read is added
   automatically), and an expiry.
2. Add it as a repository secret named `DEPENDABOT_CHANGELOG_TOKEN` under
   Settings > Secrets and variables > **Dependabot**. It must be a
   *Dependabot* secret: workflows triggered by Dependabot cannot read
   Actions secrets.
3. Put the expiry date in your calendar. When the token expires the
   workflow silently falls back to `GITHUB_TOKEN` (with the warning above)
   and the approval click comes back.

To check it works, let the next Dependabot PR open (or comment
`@dependabot recreate` on an open one). The "Updated Changelog" commit's
workflow runs should start straight away, triggered by the token's owner,
with no "Approve and run workflows" prompt.

### Security

The token is only exposed to the `changelog` job, which runs for
Dependabot-authored PRs only. That job does not run code from the PR: it
checks out the branch and runs two third-party actions, both pinned to a
commit SHA. Keep the token's scope as narrow as above.

### Approvals

If branch protection dismisses stale reviews on push, the changelog commit
could invalidate this workflow's earlier approval. `dependabot-auto-merge.yml`
re-runs on every `synchronize` event for the PR (including the one caused by
the changelog commit) and re-approves each time, so the PR is never left
stuck waiting on a dismissed review. Auto-merge itself, once enabled,
persists across new commits and does not need to be re-enabled.
