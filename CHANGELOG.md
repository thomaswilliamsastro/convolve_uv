# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Run the `lint`, `typecheck`, `docs` and `wheel` tox environments once, in a dedicated `checks` job on Ubuntu, instead of in each of the three Python 3.14 test jobs (one per OS), where they gave the same result three times. They run in parallel with the test matrix, and the required `Test` check now passes only if both the test matrix and the `checks` job do. The slowest test job, which set the time a pull request's CI takes, no longer carries them. Running `tox` locally still runs every environment (#49).
- Replace `tarides/changelog-check-action` with an equivalent inline check: the `no changelog` label skips it, otherwise `CHANGELOG.md` must differ from the base branch. The action runs an unpinned `actions/checkout` internally, which the repository's "require actions to be pinned to a full-length commit SHA" setting rejects, so the required `Check Changelog` check failed on every pull request. The inline version also fails, instead of passing, when git cannot compare the files (#48).
- Tidy the `Build` workflow: drop `submodules: true` (the repository has no submodules) and replace the `pip install .` step, which built the package a third time without affecting `python -m build`, with a step that installs the wheel that `python -m build` produced. CI still checks that the real artifact installs, dependencies included, with pip's resolver (the Test jobs use uv), without building again. The required `Build` check keeps its name (#48).
- Pin the remaining first-party GitHub Actions (`actions/checkout`, `actions/setup-python`, `actions/upload-artifact` and `actions/download-artifact`) to full commit SHAs, as the third-party ones already were, so that updates arrive as reviewable Dependabot PRs instead of through a moving tag. Each pin is the commit its tag pointed to, so behaviour is unchanged. The version comments on the other pinned actions now give the exact release (for example `# v1.14.2` instead of `# release/v1`), which is what Dependabot reads when it proposes an update (#47).
- Limit Dependabot auto-merge to patch and minor updates of Python development dependencies, instead of every Dependabot PR (PR #30 auto-merged a major bump of a third-party action). Major updates, runtime dependencies and GitHub Actions now need a manual merge. The workflow reads the update details with `dependabot/fetch-metadata`, only trusts later commits that are Dependabot's own or change nothing but `CHANGELOG.md`, turns auto-merge off when a PR stops qualifying, and no longer approves PRs, since the ruleset requires no approvals. `dependabot.yml` now groups only the minor and patch updates of the non-runtime dependencies, so major updates arrive as separate PRs (#46).
- Make the package typed: add a `py.typed` marker (PEP 561), shipped in the wheel and sdist, so type checkers use its annotations, and type check the package strictly (and the tests more leniently) with mypy in a new `typecheck` tox environment that runs in CI. Type errors this found were fixed without changing behaviour: `do_convolution` no longer reuses `pad_x` and `pad_y` for both float sizes and integer widths, and a few return values and a test helper are annotated properly. `astropy`, `radio-beam` and `spectral-cube` ship no type information, so they are treated as untyped (#44).
- Check formatting with `ruff format --check --diff` in the `lint` tox environment, so CI now fails on unformatted code, and format the three files that had drifted. The formatting changes are cosmetic only: short calls that had been wrapped for a narrower line length now fit on one line, and trailing blank lines at the end of a file are removed. The syntax tree, including docstrings, is identical before and after (#43).
- Enable a much broader set of ruff rules (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, comprehensions, NumPy, perflint, pytest-style and ruff-specific) instead of only ruff's defaults, and let ruff infer its target Python version from `requires-python` instead of a hard-coded 3.14. Enforce Google-style docstrings with ruff's pydocstyle rules and `convention = "google"`, replacing the `numpy` convention, which did not match the docstrings and was not enforced. Missing-docstring checks are skipped for the tests and the Sphinx config. Fix what the new rules found: the `AttributeError` and `ValueError` raised from `do_convolution` for a missing beam or a too-small target beam are now chained to their cause, docstring summaries end with a period and no longer have a blank line after them, `convolve_uv` has a package docstring, and a few over-long lines and test-style issues are tidied (#42).
- Stop shipping the test suite in the wheel: only the `convolve_uv` package is installed (package discovery is now explicit, with `include-package-data = false` and `convolve_uv.tests` excluded). The tests remain in the sdist and the repository. This means `pytest --pyargs convolve_uv` no longer finds tests in an installed wheel. List `pytest >= 9.0` explicitly in the `test` extra instead of relying on it arriving through `pytest-cov`, and add a `wheel` tox environment, run in CI, that uses `check-wheel-contents` to verify the built wheel contains exactly the package files and not the tests (#41).
- Declare the license as the SPDX expression `GPL-3.0-or-later` with `license-files` (PEP 639), replacing the deprecated `license = {file = ...}` table and the `License ::` classifier. The published metadata previously embedded the full licence text in its `License` field. The license is unchanged: the expression matches the existing "GPLv3+" classifier. Raises the minimum `setuptools` to 77.0.1, the first release with PEP 639 support (#40).
- Relax the exact build-requirement pins to minimum versions (`setuptools >= 77.0.1`, `setuptools_scm >= 8.0`) and drop the unneeded `wheel` requirement. Exact pins make it harder for downstream packagers and users with constraints to build from source. These minimums are the oldest versions tested (older `setuptools` also needs the separate `wheel` package) (#39).

### Fixed

- Stop `convolve_uv` from temporarily setting `allow_huge_operations` on the cube it is given. The flag was flipped and restored around the call, so another thread using the same cube saw it change, and two overlapping calls on one cube could leave it stuck on `True`, silently disabling spectral-cube's huge-operation protection for that cube. Nothing in `convolve_uv` needs the flag: spectral-cube only enforces it in `apply_function`, which is never called, so the override is removed instead of being made thread-safe. The input is now never modified. The convolved data, beam and unit are identical to before, but the returned cube now keeps the input's `allow_huge_operations` setting instead of always being set to `True` (#51).
- Fix NaN interpolation being biased towards zero near the edges of the image. `astropy.convolution.convolve_fft` counted the zero padding around the image as valid data when interpolating, so a NaN at an edge or in a corner was replaced by a value pulled towards zero (for example 0.33 instead of 1.0 for a corner pixel of a constant image, with a kernel sigma of 2 pixels), and the convolved result was darkened around it (by 4.6% for a 2 x 2 block of NaNs in a corner, with a 1.5" target beam). The padding is now ignored, so the interpolated values equal a plain kernel-weighted mean of the valid pixels. A NaN with no valid data within reach of the kernel (about five kernel widths) used to be replaced by 0 and then treated as real data; it now stays NaN and is excluded from the convolution. Results for images with NaNs near an edge, or with large NaN holes, will change (#50).
- Push the Dependabot changelog commit with a token from the `DEPENDABOT_CHANGELOG_TOKEN` Dependabot secret instead of `GITHUB_TOKEN`. A commit pushed by `github-actions[bot]` had all its CI runs held at `action_required` until a maintainer approved them, so the required checks never finished and auto-merge stalled on every Dependabot PR. Without the secret the workflow falls back to `GITHUB_TOKEN` with a warning. Document the setup, and correct the description of how the workflows interact, in `.github/DEPENDABOT_AUTO_MERGE.md` (#45).
- Raise a clear `ValueError` instead of attempting an enormous allocation when the padded image (for a huge `pad_sigma`, or a target beam far wider than the pixel scale) or the NaN interpolation kernel would exceed 2**28 pixels per array. Previously such requests ran effectively forever or exhausted memory (#38).
- Read `convolve_uv.__version__` from the installed package metadata instead of a generated `convolve_uv/version.py`, which no longer needs to be written. The old import left the fallback branch uncovered whenever a build had generated that file, so coverage was incomplete; `__init__.py` is now fully covered, with tests for both the installed and uninstalled cases (#37).

## [0.4.0] - 2026-09-30

### Added

- Warn with `convolve_uv.LargeCubeMemoryWarning` when convolving a cube large enough that spectral-cube's `allow_huge_operations` safeguard would normally apply, and document that `convolve_uv` materializes the whole cube into memory (#28).
- Add a `dependabot-auto-merge.yml` GitHub Actions workflow that approves Dependabot pull requests and enables auto-merge once all required checks pass, without bypassing branch protection or using `pull_request_target` (#29).

### Fixed

- Exclude masked pixels from convolution weights and preserve input data types (#14).
- Correct parameter documentation and make usage examples copy-pasteable (#16).
- Restore `allow_huge_operations` after `convolve_uv` instead of permanently mutating caller state (#17).
- Build the NaN interpolation kernel from the full pixel-space beam covariance instead of a single, isotropic WCS pixel scale, fixing incorrect interpolation for anisotropic/rotated pixels (#18).
- Harden invalid-input and NaN-handling tests, with full line and branch coverage for `convolve_uv/convolve_uv.py` (#19).
- Validate finite, non-negative padding and report contextual errors for invalid WCS and covariance matrices (#25).
- Make varying-resolution test beam orientations deterministic without mutating global random state (#26).
- Select the `-cov` tox environments in the `tox-gh` mapping, and fail the test job if the coverage or JUnit reports are missing (#31).
- Install `tox-uv` alongside `tox-gh` in CI so each test job runs only the tox environments for its own Python version, instead of every job re-provisioning tox and running the entire environment list (#33).
- Gate the PyPI upload on the full test matrix, run as a reusable workflow on release tags, and fail the release build if the tag does not match the built package version (#34).
- Derive the package version from the latest `v*` git tag with `setuptools_scm` instead of a hard-coded value in `pyproject.toml`, so `convolve_uv.__version__` and the documentation's version are correct (the former was always `"dev"`) and releasing is just tagging. The generated `convolve_uv/version.py` is git-ignored, builds without git metadata fall back to `0.0.0+unknown`, and the build and publish workflows check out full history (#35).

### Updated

- Ignore tox-generated coverage and JUnit report files (#15).
- Harden GitHub Actions workflow and Dependabot security: least-privilege permissions, PyPI Trusted Publishing (OIDC), SHA-pinned third-party actions, and consistent job/step naming (#20).
- Document core dependency minimums, test minimum and latest compatible dependency sets in CI for every supported Python version, and separate core runtime Dependabot updates (#23).
- Split numerical convolution helpers (beam/pixel covariance, transfer functions, FFT filtering, interpolation kernel construction, 2D convolution) into a new `convolve_uv._numerics` module, separate from the `convolve_uv()` orchestration over `SpectralCube`/`Projection`. The behavior and signatures of the `convolve_uv()` entry point and of the moved helpers are unchanged, as are the package exports (`convolve_uv` and `LargeCubeMemoryWarning`). **Breaking for deep imports:** the helpers (`beam_covariance_en`, `kernel_covariance_pixels`, `nan_interpolation_kernel`, `transfer_function`, `fft_filter` and `FWHM_TO_SIGMA`) are no longer importable from `convolve_uv.convolve_uv`; import them from `convolve_uv._numerics` instead (#32).

### Dependencies
- Bump `setuptools-scm` from 10.2.3 to 10.3.4 ([#24](https://github.com/thomaswilliamsastro/convolve_uv/pull/24))
- Bump `astral-sh/setup-uv` from 7.6.0 to 10.2.0 ([#30](https://github.com/thomaswilliamsastro/convolve_uv/pull/30))

## [0.3.0]

### Added

- Added documentation (#12).

### Updated

- Bump version to 0.3.0 (#13).

## [0.2.0]

### Added

- Added comprehensive test suite (#10).

### Updated

- Bump version to 0.2.0 (#11).

## [0.1.1]

### Fixed

- Updated badges in README.md (#6).

### Updated

- Updated a number of dependencies.

## [0.1.0]

### Added

- Initial release (#1).
