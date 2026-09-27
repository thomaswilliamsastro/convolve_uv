# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Exclude masked pixels from convolution weights and preserve input data types (#14).
- Correct parameter documentation and make usage examples copy-pasteable (#16).
- Restore `allow_huge_operations` after `convolve_uv` instead of permanently mutating caller state (#17).
- Build the NaN interpolation kernel from the full pixel-space beam covariance instead of a single, isotropic WCS pixel scale, fixing incorrect interpolation for anisotropic/rotated pixels (#18).
- Harden invalid-input and NaN-handling tests, with full line and branch coverage for `convolve_uv/convolve_uv.py` (#19).
- Validate finite, non-negative padding and report contextual errors for invalid WCS and covariance matrices (#25).
- Make varying-resolution test beam orientations deterministic without mutating global random state (#26).

### Updated

- Ignore tox-generated coverage and JUnit report files (#15).
- Harden GitHub Actions workflow and Dependabot security: least-privilege permissions, PyPI Trusted Publishing (OIDC), SHA-pinned third-party actions, and consistent job/step naming (#20).
- Document core dependency minimums, test minimum and latest compatible dependency sets in CI for every supported Python version, and separate core runtime Dependabot updates (#23).

### Dependencies

- Bump `setuptools-scm` from 10.2.3 to 10.3.4 ([#24](https://github.com/thomaswilliamsastro/convolve_uv/pull/24))

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
