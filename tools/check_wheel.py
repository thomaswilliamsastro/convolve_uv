"""Check a built wheel against what ``pyproject.toml`` promises.

``check-wheel-contents`` compares the files in a wheel with the package, but it does not
look at the metadata. Two things it cannot catch:

* the license: a wheel without the license text still passes, even though its metadata
  declares the license and the project is distributed under it;
* the version: ``convolve_uv.__version__`` is read from the installed metadata under the
  name ``convolve_uv``, and quietly becomes ``"dev"`` if the wheel is published under
  another name, or its version is the placeholder used when git metadata is missing.

The expected values are read from ``pyproject.toml``, not written down again here.

Usage: ``python tools/check_wheel.py DIST_DIR``, where DIST_DIR holds exactly one wheel.
"""

import sys
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from importlib.metadata import Distribution
from pathlib import Path

PACKAGE = "convolve_uv"
ROOT = Path(__file__).resolve().parent.parent


def check(wheel_path: Path) -> tuple[list[str], str]:
    """Return a message for everything wrong with the wheel, and a summary of what was checked."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    problems: list[str] = []
    summary = ""

    with zipfile.ZipFile(wheel_path) as wheel, tempfile.TemporaryDirectory() as unpacked:
        names = wheel.namelist()
        dist_infos = sorted({name.split("/")[0] for name in names if ".dist-info/" in name})
        if len(dist_infos) != 1:
            return [f"expected one .dist-info directory in the wheel, found {dist_infos}"], summary
        dist_info = dist_infos[0]
        for name in names:
            if name.startswith(dist_info + "/"):
                wheel.extract(name, unpacked)
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode())

        # The license: declared in the metadata, and the text itself shipped in the wheel
        if metadata["License-Expression"] != project["license"]:
            problems.append(
                f"License-Expression is {metadata['License-Expression']!r}, "
                f"pyproject.toml has {project['license']!r}"
            )
        expected = sorted(
            {path for pattern in project["license-files"] for path in ROOT.glob(pattern)}
        )
        if not expected:
            problems.append("the license-files patterns in pyproject.toml match no file")
        declared = metadata.get_all("License-File") or []
        for path in expected:
            relative = path.relative_to(ROOT).as_posix()
            shipped = f"{dist_info}/licenses/{relative}"
            if shipped not in names:
                problems.append(f"{relative} is not in the wheel (looked for {shipped})")
            elif wheel.read(shipped) != path.read_bytes():
                problems.append(f"{relative} in the wheel differs from the one in the repository")
            if relative not in declared:
                problems.append(f"the metadata has no License-File entry for {relative}")

        # The version: what the package finds when it asks for its own installed version
        found = list(Distribution.discover(name=PACKAGE, path=[unpacked]))
        if not found:
            problems.append(
                f"the wheel's metadata is not found under the name {PACKAGE!r} "
                f"(its name is {metadata['Name']!r}), so __version__ would be 'dev'"
            )
        else:
            fallback = pyproject["tool"]["setuptools_scm"]["fallback_version"]
            if found[0].version == fallback:
                problems.append(
                    f"the version is the placeholder {fallback!r}, "
                    "used when git metadata is missing"
                )
            summary = (
                f"{wheel_path.name}: version {found[0].version}, "
                f"license files {[path.name for path in expected]}"
            )
    return problems, summary


def main() -> int:
    """Check the one wheel in the directory given on the command line."""
    wheels = sorted(Path(sys.argv[1]).glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected exactly one wheel in {sys.argv[1]}, found {len(wheels)}")
        return 1
    problems, summary = check(wheels[0])
    for problem in problems:
        print(f"ERROR: {problem}")
    if not problems:
        print(summary)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
