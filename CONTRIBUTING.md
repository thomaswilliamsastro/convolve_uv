# Contributing to convolve_uv

Thank you for your interest in `convolve_uv`. Bug reports, questions, documentation fixes and pull
requests are all welcome.

## Reporting a problem or asking a question

Open an [issue](https://github.com/thomaswilliamsastro/convolve_uv/issues). For a bug, please include:

- the versions of `convolve_uv`, `astropy`, `spectral-cube` and `radio-beam` (`pip list`), and of Python;
- what you expected and what happened, with the full error message;
- a small example that shows it. One that builds a small cube in code is much easier to work with
  than one that needs a large FITS file.

To report a security vulnerability, please do not open an issue. See [SECURITY.md](SECURITY.md).

## Setting up

You need Python 3.12 or newer. Fork and clone the repository, then install it in editable mode with the
test tools and run the tests:

```bash
pip install -e ".[test]"
pytest
```

The other checks are [tox](https://tox.wiki/en/latest/) environments. `tox.ini` uses the `tox-uv` plugin, so install
both:

```bash
pip install tox tox-uv
```

| Command | What it does |
|---|---|
| `tox -e py312` | Runs the tests. Use `py313` or `py314` for another Python. |
| `tox -e py312-cov` | The same, with coverage and the coverage and JUnit reports. |
| `tox -e py312-oldestdeps` | The same, with the lowest versions of the dependencies that `pyproject.toml` allows. |
| `tox -e lint` | `ruff check` and `ruff format --check`. |
| `tox -e typecheck` | `mypy`, strict for the package and more lenient for the tests. |
| `tox -e docs` | Builds the documentation, checks its links and its coverage. Warnings are errors. |
| `tox -e wheel` | Builds the wheel and checks its contents and metadata. |
| `tox` | Everything above, for every supported Python. It takes a while. |

`lint`, `typecheck`, `docs` and `wheel` also have `-oldestdeps` variants (for example
`tox -e typecheck-oldestdeps`), which CI runs too. They check that the minimum versions the package
declares really work.

tox keeps each environment in `.tox/`. They take from about 10 to 300 MB each, a few GB for all of them, and
git ignores them, as it does the coverage and JUnit reports that the `-cov` environments write to the
repository root. Delete `.tox/` to reclaim the space; tox recreates what it needs.

To fix formatting rather than just report it, install the linter with `pip install -e ".[test-ruff]"` and run
`ruff format` and `ruff check --fix`.

## Code and tests

- Docstrings use the Google style, which `ruff` enforces. The line length is 100.
- The package is fully type annotated and checked with `mypy --strict`.
- Tests are in `convolve_uv/tests/`. `test_convolve_uv.py` covers the public `convolve_uv` function,
  `test_numerics.py` covers the helpers in `convolve_uv/_numerics.py`, and `helpers.py` holds the builders
  they share.
- The project has 100% line and branch coverage. Please keep new code covered, and add a test that fails
  without your change when you fix a bug.
- The public API is `convolve_uv` and `LargeCubeMemoryWarning`, both imported from the package. Everything
  else, including `convolve_uv._numerics`, is internal and can change without notice.

## Documentation

The sources are in `docs/` and are built with Sphinx and hosted on
[Read the Docs](https://convolve-uv.readthedocs.io). Check changes with `tox -e docs`.

## Changelog

A pull request that changes something users can see needs an entry in `CHANGELOG.md`, under
`## [Unreleased]` in the section that fits (Added, Changed, Fixed and so on). The required
`Check Changelog` check fails without one. End the entry with a link to the pull request:

```markdown
- Fix the thing that was broken ([#123](https://github.com/thomaswilliamsastro/convolve_uv/pull/123)).
```

You only know the number once the pull request is open, so add it in a second push. If the change is not
user-visible, ask a maintainer to add the `no changelog` label. Dependabot's entries are added automatically.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- The required checks are `Build`, `Test` and `Check Changelog`. `Test` runs the tests on Linux, macOS and
  Windows with Python 3.12 to 3.14, and the lint, typecheck, docs and wheel checks, each also with the
  oldest dependencies.
- Pull requests are merged by squashing, so each becomes a single commit on `main` titled after the pull
  request. A clear title and description are what end up in the history.

## Releases

The maintainer makes releases by tagging `main`. The steps, and the one-off PyPI setup behind them, are in
[`.github/RELEASING.md`](.github/RELEASING.md).

## Licence

`convolve_uv` is distributed under the [GPL-3.0-or-later](LICENSE) licence. By contributing, you agree that
your contribution is licensed under the same terms.
