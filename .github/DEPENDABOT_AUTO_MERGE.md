# Dependabot auto-merge

`.github/workflows/dependabot-auto-merge.yml` enables GitHub's native
auto-merge for Dependabot pull requests that are **low-risk**, and makes sure
it is off for every other one. It never merges anything itself and never
bypasses the `main` ruleset: GitHub performs the actual merge, and only once
**all** required status checks pass. Everything it does not enable stays a
normal PR for a maintainer to review and merge.

## What gets auto-merged

Only **patch and minor updates of Python development dependencies**, and only
if *every* dependency in the PR qualifies:

| Update | Auto-merged? | Why |
|---|---|---|
| Patch/minor of a development dependency (pytest, ruff, mypy, sphinx, setuptools, setuptools_scm, ...) | yes | Only affects CI and the build; the required checks cover it |
| Any **major** update | no | Can break things in ways the tests do not cover |
| Runtime dependencies (astropy, numpy, radio-beam, spectral-cube) | no | For a library, raising a dependency's minimum version changes what its users can install alongside it |
| GitHub Actions, including patch/minor | no | Third-party actions are pinned to commit SHAs so that updates are a deliberate choice; auto-merging the bump would defeat that (a compromised release would run in CI and, at release time, next to the PyPI publish step) |

History shows why the gate exists: PR #30 bumped `astral-sh/setup-uv` from 7.6.0
to 10.2.0, a **major** update of a third-party action, and went through the
previous workflow automatically.

`dependabot.yml` puts major updates of the non-runtime dependencies in their
own PRs (only minor and patch updates are grouped), so one major update cannot
hold back the minor and patch updates that are eligible.

## How each run decides

Every run (when a PR is opened, reopened or pushed to, by anyone) does this and
writes the verdict and a table of the dependencies to the job summary:

1. **Metadata.** `dependabot/fetch-metadata` reads the update details (package
   ecosystem, dependency type, update type) from Dependabot's *first* commit,
   which it checks is Dependabot's own and signed.
2. **Policy.** Every dependency must be a `pip` development dependency with a
   patch or minor update. If the metadata cannot be read, the PR is not
   eligible.
3. **Later commits.** `fetch-metadata` does not look beyond the first commit,
   and auto-merge merges whatever the branch head is, so this workflow checks
   them itself: each later commit must be either a signed Dependabot commit or
   change nothing but `CHANGELOG.md` (the changelog bot's commit). Any other
   commit, for example a stray push to the branch, makes the PR ineligible.

An eligible PR gets `gh pr merge --auto --squash`. For an ineligible one, the
workflow turns auto-merge **off** if it was on, for example because Dependabot
later added a major update to a group, or an unexpected commit appeared.

To change the policy, edit the `Decide whether the update is low-risk` step.

## Why `pull_request`, not `pull_request_target`

The workflow triggers on `pull_request`, never checks out the PR's code,
and never runs anything from the PR diff - it only calls the GitHub API
via `gh` using trusted event metadata (`github.event.pull_request.*`).
This avoids the classic `pull_request_target` + "checkout untrusted PR
head" privilege-escalation pattern, at the cost of nothing here, since the
job doesn't need to build or test the code (that's what `tests.yml` /
`build.yml` are for).

## Required repository settings

- **Settings → General → Pull Requests → "Allow auto-merge"** must be
  enabled for the repository. Without this, `gh pr merge --auto` fails.
- **The `main` ruleset** must require the status checks this repository
  already runs on pull requests (`Test`, `Build`, `Check Changelog`), so
  GitHub's auto-merge cannot complete until they pass.
- The ruleset currently requires **0 approving reviews**, which is why this
  workflow no longer approves anything (the old "approve" step was a no-op). If
  you ever require reviews, auto-merge will wait for a human approval.
- **"Allow GitHub Actions to create and approve pull requests"** (Settings →
  Actions → General) is no longer needed by this workflow and can be turned
  off.

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

### The decision and the changelog commit

The changelog commit arrives after Dependabot's own, and pushing it (with the
token above) starts this workflow again. That is expected: it is the kind of
later commit the workflow allows (it changes only `CHANGELOG.md`), so the PR
stays eligible and auto-merge stays on. Auto-merge, once enabled, persists
across new commits and does not need to be re-enabled.
