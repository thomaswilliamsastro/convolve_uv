# Releasing convolve_uv

A release is a git tag on `main`. The tag is the only place the version is set: `setuptools_scm` reads it
when the package is built, so there is no version number to edit in the code. Between releases, `main`
builds as a development version such as `0.4.1.dev33+g4459db6`.

Pushing a tag that starts with `v` runs the `Publish` workflow
([`publish.yml`](workflows/publish.yml)). It checks that the tagged commit is on `main`, runs the full test
workflow, builds the source distribution and the wheel, checks that the tag matches the version of what it
built, installs each of the two files into a clean environment and uses it (`tools/check_install.py`), and then
uploads both to PyPI with trusted publishing, so there is no API token. The upload waits for your approval
(step 3). Nothing else is automatic: the GitHub release is written by hand (step 4).

Versions follow [Semantic Versioning](https://semver.org), tagged `vMAJOR.MINOR.PATCH`. While the major
version is 0, a breaking change may come in a minor release. Mark it in the changelog with an entry that
starts with **Breaking:**, as for [#56](https://github.com/thomaswilliamsastro/convolve_uv/pull/56).

## Before you start

- `main` is green, and nothing that belongs in the release is still open. The required checks are `Build`,
  `Test` and `Check Changelog`.
- Read `## [Unreleased]` in `CHANGELOG.md`. Every change users can see since the last tag should be there,
  and breaking changes should be marked. Dependabot adds its own entries under `### Dependencies`.
- `CITATION.cff` is complete: the title, the abstract, the authors with their affiliation and ORCID iD, and the
  keywords. Zenodo builds the permanent record of the release from it. `tox -e wheel` validates the file and
  compares it with `pyproject.toml`. Do not add a `.zenodo.json`, which would override `CITATION.cff`.

## 1. Cut the changelog

Open a pull request that changes only these three things:

1. In `CHANGELOG.md`, put a new, empty `## [Unreleased]` heading above the existing entries, and turn the
   old `## [Unreleased]` heading into `## [X.Y.Z] - YYYY-MM-DD`, with the date you will push the tag. Keep an
   `## [Unreleased]` heading at the top: the Dependabot changelog bot adds to it.
2. At the bottom of `CHANGELOG.md`, point `[Unreleased]` at `compare/vX.Y.Z...HEAD` and add a new
   `[X.Y.Z]: https://github.com/thomaswilliamsastro/convolve_uv/compare/vPREVIOUS...vX.Y.Z` line above the
   previous version's.
3. In `CITATION.cff`, set `date-released` to the same date, in quotes. Only its year shows in a citation.

The 0.4.0 release was prepared this way in [#36](https://github.com/thomaswilliamsastro/convolve_uv/pull/36).
Merge the pull request like any other: pull requests are squash-merged, which is the only method the
repository allows.

## 2. Tag the release commit

```bash
git switch main
git pull --ff-only
git log -1                # must be the commit from step 1
git tag vX.Y.Z
git push origin vX.Y.Z
```

Only tag a commit that is on `main`. The workflow refuses any other: the build job fails with "is not on
main" and nothing is published. That guards against a mistake, not against someone who edits the workflow
in the tagged commit, since a workflow runs from its own copy in that commit. What stops that is the
`release tags` ruleset (see the setup at the end): only its bypass actor, the repository owner, can create,
move or delete a `v*` tag. It does not stop the owner. The `pypi` environment only accepts a tag that starts
with `v`, and then waits for an approval before the upload.

## 3. Watch the publish

Open the `Publish` run for the tag in the Actions tab. Its jobs are `Build source distribution` (the tag
checks, the build, `twine check` and the smoke test of the wheel and the sdist), `Run tests` (the whole test
workflow) and `Upload to PyPI`, which needs the other two. The upload does not start unless all of that
passes.

When it has all passed, the run stops at `Upload to PyPI` with "Waiting for review", because the `pypi`
environment requires your approval. This is the last look before something that cannot be undone: PyPI
never lets a file name be used twice. Check that the version the smoke test printed in the build job (for
example `convolve_uv 0.5.0 from ...`) is the one you meant, then choose **Review deployments**, tick `pypi`
and choose **Approve and deploy**. Choosing **Reject** instead uploads nothing; then fix the problem as
under *If something goes wrong*.

To rehearse everything except the release, choose **Run workflow** for `Publish` on a branch (for example
the release-prep branch, before merging it). That builds, checks and smoke-tests the files and publishes
nothing, because only a run on a `v*` tag runs the tests and the upload.

## 4. Create the GitHub release

When the upload has finished, write the release from the changelog section, as for earlier releases:

```bash
awk -v v="X.Y.Z" '/^## \[/{p=(index($0,"[" v "]")==4); next} p' CHANGELOG.md > notes.md
gh release create vX.Y.Z --verify-tag --title vX.Y.Z --notes-file notes.md
rm notes.md
```

Publishing the GitHub release is also what creates the Zenodo record: the webhook listens for releases, not
for tags or for the PyPI upload. So do it when you are ready for a record that is hard to remove.

## 5. Check it

- PyPI shows the new version at <https://pypi.org/project/convolve-uv/>. In a fresh environment,
  `pip install convolve-uv==X.Y.Z` and `python -c "import convolve_uv; print(convolve_uv.__version__)"` should
  print the version.
- On [Read the Docs](https://convolve-uv.readthedocs.io), `latest` is built from `main` and every tag gets a
  version of its own, with `stable` following the newest. Check that the build for the tag passed.
- The GitHub release is marked **Latest**.
- On [Zenodo](https://zenodo.org/search?q=convolve_uv) a record for the release appears, described from
  `CITATION.cff`, with the tag as its version. The first time, copy the DOI for *all versions* (the concept
  DOI) into `CITATION.cff` as `doi:` and add a DOI badge to the README, in a pull request after the release.
  It cannot be added earlier, because the DOI does not exist until the first record does.

## If something goes wrong

- **The tests, a tag check or the smoke test fail.** Nothing was uploaded. Fix the problem with a pull
  request. Then delete the tag (`git push --delete origin vX.Y.Z` and `git tag -d vX.Y.Z`) and tag the fixed
  commit again.
- **The upload fails** (for example because the trusted publisher on PyPI does not match). Fix the cause,
  then use **Re-run failed jobs** on the same run (approve the deployment again if it asks). If that is no
  longer possible, choose **Run workflow**
  for `Publish` and pick the tag `vX.Y.Z` as the ref to run from: that repeats the whole release run,
  tests and upload included. Either way PyPI refuses files it already has, so this is only for an upload
  that did not complete.
- **A bad release was published.** PyPI does not let a file name be used twice, so a published version
  cannot be replaced. Yank it on PyPI (the project's *Manage* page, then the release) and publish a new
  patch release.

## One-off setup

### The trusted publisher on PyPI

For the project `convolve-uv` on PyPI, under *Publishing*, add a GitHub trusted publisher with:

| Field | Value |
|---|---|
| Owner | `thomaswilliamsastro` |
| Repository name | `convolve_uv` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

PyPI matches on these, so the workflow's file name and the environment name in `publish.yml` have to stay as
they are, or be changed on PyPI at the same time.

### The Zenodo integration

On Zenodo, under *GitHub*, the switch for `thomaswilliamsastro/convolve_uv` is on. That added a webhook to
the repository that listens for `release` events. Zenodo only archives releases published after the switch
was turned on, so v0.4.0 and earlier have no record.

### The `release tags` ruleset on GitHub

Under *Settings*, *Rules*, *Rulesets*, the ruleset `release tags` targets tags matching `refs/tags/v*`. It
restricts creating, updating and deleting them, and its only bypass actor is the repository owner, set up
the same way as on the `main` ruleset, so that the owner can still tag a release, and delete a tag made too
early (see *If something goes wrong*). Deleting the ruleset removes the protection.

### The `pypi` environment on GitHub

The repository has an environment called `pypi`. Only tags matching `v*` can deploy to it, and each
deployment needs approval from a required reviewer, the repository owner. Self-review is allowed, because
the owner is the only collaborator: with it blocked nobody could approve a release. Administrators can
bypass the protection rules. This is a deliberate pause before an upload that cannot be undone, not a
security boundary: any token that can push a tag could also approve as the owner. To remove it, clear the
reviewers under *Settings*, *Environments*, `pypi`.
