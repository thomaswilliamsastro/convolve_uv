"""Check that CITATION.cff agrees with pyproject.toml.

GitHub builds its "Cite this repository" entry from CITATION.cff, and Zenodo builds the permanent
record of every release from it. Both quietly ignore a file that is not valid (``cffconvert
--validate`` checks that), and neither can tell that the keywords or the license here have drifted
away from the ones on PyPI. This compares what the two files both state, so they are written down
once in pyproject.toml and checked here, not trusted to be kept in step.

Usage: ``python tools/check_citation.py``, from anywhere. It needs PyYAML, which the ``wheel`` tox
environments install.
"""

import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def check() -> list[str]:
    """Return a message for everything in CITATION.cff that disagrees with pyproject.toml."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    problems: list[str] = []

    def compare(what: str, citation_value: object, project_value: object) -> None:
        if citation_value != project_value:
            problems.append(
                f"{what}: CITATION.cff has {citation_value!r}, pyproject.toml has {project_value!r}"
            )

    compare("keywords", sorted(citation.get("keywords") or []), sorted(project["keywords"]))
    compare("license", citation.get("license"), project["license"])
    compare("title", citation.get("title"), project["name"])
    compare("repository-code", citation.get("repository-code"), project["urls"]["Source"])
    author = (citation.get("authors") or [{}])[0]
    compare(
        "first author",
        f"{author.get('given-names')} {author.get('family-names')}",
        project["authors"][0]["name"],
    )
    return problems


def main() -> int:
    """Print what disagrees, and return 1 if anything does."""
    problems = check()
    for problem in problems:
        print(f"ERROR: {problem}")
    if not problems:
        print(
            "CITATION.cff agrees with pyproject.toml (keywords, license, title, repository, author)"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
