# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add a `show_progress` keyword to `convolve_uv` (default `True`, so nothing changes unless it is used). `show_progress=False` draws no progress bar when convolving a cube. The bar was always created, and on a terminal it installs a `SIGWINCH` signal handler, which Python only allows in the main thread, so calling `convolve_uv` on a cube from a worker thread (a thread pool, a web server) failed with `ValueError: signal only works in main thread of the main interpreter`. Off a terminal the bar was already silent, so the failure only appeared when the same code was run interactively. The keyword is ignored for a `Projection`, which has no bar (#52).

### Changed

- Split the test suite to match the module split: the tests of the numerical helpers in `convolve_uv._numerics` (covariance, the NaN interpolation kernel and edges, and `do_convolution` called directly) move to `tests/test_numerics.py`, the builders both files use move to `tests/helpers.py`, and `tests/test_convolve_uv.py` keeps the tests of the public `convolve_uv` function and of the checks shared with `do_convolution`. It had grown to about 1,700 lines in one file and one class. This only moves tests, none was changed: all 153 test IDs are the same apart from their file and class, and every test function and helper has an identical syntax tree before and after. Coverage is unchanged (#61).
- Stop publishing the private `convolve_uv._numerics` helpers in the API reference, which now documents exactly what the package exports (`convolve_uv` and `LargeCubeMemoryWarning`) and says that everything else is internal and may change without notice. The page had published `beam_covariance_en`, `kernel_covariance_pixels`, `nan_interpolation_kernel`, `transfer_function`, `fft_filter` and `do_convolution` with `:undoc-members:`, so internals read as an API users could depend on, and any new helper would have been published automatically. Nothing in the README or the other pages used them. The helpers themselves are unchanged and still importable from `convolve_uv._numerics`; only the documentation changed, so links to their old `convolve_uv._numerics.*` anchors no longer resolve (#59).
- Check the built wheel's metadata, which `check-wheel-contents` does not look at, with a new `tools/check_wheel.py` run by the `wheel` tox environments. The wheel must ship the license text named by `license-files` (unchanged from the repository's copy) and list it in its metadata, `License-Expression` must match `pyproject.toml`, and `convolve_uv.__version__` must resolve: the wheel's metadata must be found under the name `convolve_uv` and its version must not be the `0.0.0+unknown` placeholder used when git metadata is missing. All three were possible to break while the `wheel` environment still passed: `license-files = []`, or a pattern that matches nothing, built a wheel with no license text but a declared license, and a project name that differs from `convolve_uv` built a wheel that installs with `__version__ == "dev"`. The expected values are read from `pyproject.toml`, so they are not written down twice. Nothing was wrong in a release: v0.4.0 ships its license text and a resolvable version (its metadata predates the `License-Expression` field of the PEP 639 change), so this guards against a future change rather than fixing a published one (#58).
- Check out the full history and tags in the test jobs and the checks job of `tests.yml`, as the build, publish and changelog jobs already do. They used a shallow checkout, so `setuptools_scm` could not find the version and every package under test was built as `0.0.1.dev1+unknown` (the CI logs show it in the test jobs), which also hid whether version handling worked in the jobs that run the tests. They now build a real development version such as `0.4.1.dev24+g<commit>`. The tests do not depend on the version, so this changes no results (#58).
- Check that the minimum versions declared in `pyproject.toml` work. Only the runtime dependencies and the `test` extra had a lowest-version run (the `oldestdeps` test jobs); the build requirements and the `lint`, `typecheck` and `docs` tools were only ever tried at their newest versions, so a floor could stop working unnoticed. `lint`, `typecheck`, `docs` and `wheel` now have `-oldestdeps` variants, run in CI in a second, parallel `Checks (oldest dependencies)` job next to the existing one (the required `Test` check needs both). The first three use `uv`'s lowest-direct resolution. `wheel-oldestdeps` installs the `[build-system]` requirements at their lowest versions itself and builds without build isolation, because `uv` does not lower a build's own requirements; it reads them from `pyproject.toml`, so the floors are not written down twice. All floors held (`setuptools` 77.0.1, `ruff` 0.15.10, `mypy` 1.19.0, `sphinx` 9.1.0 and the rest), and lowering a floor to a version that does not work makes the matching job fail. One declared floor was not installable: `setuptools_scm` 8.0.0 is yanked from PyPI, so the floor is now 8.0.1, which no resolver could have gone below anyway (#57).
- **Breaking:** move the contents of the `convolve_uv.convolve_uv` module to a private `convolve_uv._convolve` module, so `convolve_uv.convolve_uv` is only the function. It used to be both: the package re-exports the function under the name of the submodule that defines it, so `import convolve_uv.convolve_uv as m` silently gave the function, `monkeypatch.setattr("convolve_uv.convolve_uv.name", ...)` failed, and Sphinx cross-references to the name resolved to the module instead of the function. Use `from convolve_uv import convolve_uv, LargeCubeMemoryWarning`, which is what the documentation already shows and still works. `from convolve_uv.convolve_uv import ...` now raises `ModuleNotFoundError`, and pickles of the function made with an earlier version no longer load. The API reference now documents the top-level names. Behaviour is otherwise unchanged (#56).
- Document that an image which already has the target beam is returned unchanged, so with the default `nan_treatment='interpolate'` its `NaN` values stay `NaN`, while they are interpolated in every other channel. This is the case for the widest channel when convolving a varying-resolution cube to its common beam. The behaviour is not new: it was only described on the private `do_convolution`, not on `convolve_uv` or in the usage page, which now explain it and how to get the same treatment in every channel (`nan_treatment='fill'` or `preserve_nan=True`). Add tests for masked input on this path, which treats masked pixels as missing exactly as the convolution does, and for the mixed-channel behaviour. No behaviour changes (#55).
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

- Raise a clear `TypeError` when a cube is passed to `do_convolution`, which handles one 2D image at a time: "image_slice must be a single 2D image, such as a channel slice cube[0], but it has 3 dimensions. To convolve a full cube, use convolve_uv". It used to fail with an error that depended on the kind of cube: `ValueError: too many values to unpack`, `AttributeError: 'VaryingResolutionSpectralCube' object has no attribute 'beam'`, or a misleading "image_slice must have a valid beam". `convolve_uv` is unaffected, as it passes each channel as a 2D slice. `do_convolution` is internal (`convolve_uv._numerics`), so the new error only changes what someone calling it directly sees (#60).
- Check every argument of `convolve_uv` before it touches the cube. `nan_treatment` and the type of `target_beam` were only checked inside `do_convolution`, once per channel, so a bad value only raised after `convolve_uv` had allocated a full-size output array, copied the cube, started the progress bar and, for a large cube, emitted the `LargeCubeMemoryWarning` for a call that then failed. The errors and their messages are unchanged; they now come first. The `boundary` and `pad_sigma` checks, which were written out in both `convolve_uv` and `do_convolution`, and the new ones now live in one helper that both use, so the two entry points cannot drift apart (#54).
- Correct the `do_convolution` docstring, which said it accepted a full `SpectralCube` ("if a full SpectralCube, then the cube should only have two dimensions"). It only accepts a single 2D image, such as a channel slice (`cube[0]`) or a `Projection`; passing a cube, even one with a single channel, fails with `ValueError: too many values to unpack`. The docstring now says so and points to `convolve_uv` for cubes. No behaviour changes (#53).
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
