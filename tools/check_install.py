"""Smoke test an installed convolve_uv: check what was installed and convolve a small cube.

The publish workflow installs the wheel and the source distribution it is about to upload, each into
a clean environment, and runs this with that environment's Python. It checks that the files really
install and work, which the test suite, run against the checkout, does not.

Usage: ``python -I tools/check_install.py EXPECTED_VERSION``. The ``-I`` keeps the checkout out of
``sys.path``, so that the installed package is the one imported.
"""

import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from radio_beam import Beam
from spectral_cube import Projection, SpectralCube

import convolve_uv

PIXEL_SCALE = 1 * u.arcsec
# Big enough that the edges are many beam widths from the source: near an edge the default 'fill'
# boundary renormalises, which would make a comparison with the analytic kernel needlessly loose
SIZE = 101
INPUT_BEAM = Beam(major=3 * u.arcsec, minor=2 * u.arcsec, pa=20 * u.deg)
TARGET_BEAM = Beam(major=8 * u.arcsec, minor=8 * u.arcsec, pa=0 * u.deg)


def make_cube() -> SpectralCube:
    """Return a small cube of the input beam's own shape, so convolving gives the target beam."""
    kernel = INPUT_BEAM.as_kernel(pixscale=PIXEL_SCALE, x_size=SIZE, y_size=SIZE).array
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
    wcs.wcs.crval = [0.0, 0.0, 0.0]
    wcs.wcs.cdelt = [-PIXEL_SCALE.to_value(u.deg), PIXEL_SCALE.to_value(u.deg), 2500]
    wcs.wcs.crpix = [0.0, 0.0, 0.0]
    data = np.ones((3, 1, 1)) * kernel[np.newaxis] * u.K
    return SpectralCube(data=data.astype(np.float32), wcs=wcs, beam=INPUT_BEAM)


def check(expected_version: str) -> list[str]:
    """Return a message for everything wrong with the installation, or an empty list."""
    problems: list[str] = []

    location = Path(convolve_uv.__file__).resolve()
    if "site-packages" not in location.parts:
        problems.append(f"convolve_uv was imported from {location}, not from an installation")
    if convolve_uv.__version__ != expected_version:
        problems.append(
            f"__version__ is {convolve_uv.__version__!r}, expected {expected_version!r}"
        )
    if sorted(convolve_uv.__all__) != ["LargeCubeMemoryWarning", "convolve_uv"]:
        problems.append(f"the public names are {sorted(convolve_uv.__all__)}")
    if not issubclass(convolve_uv.LargeCubeMemoryWarning, Warning):
        problems.append("LargeCubeMemoryWarning is not a warning class")

    expected = TARGET_BEAM.as_kernel(pixscale=PIXEL_SCALE, x_size=SIZE, y_size=SIZE).array
    cube = make_cube()
    result = convolve_uv.convolve_uv(cube, TARGET_BEAM, show_progress=False)
    if not isinstance(result, SpectralCube) or result.beam != TARGET_BEAM:
        problems.append("convolving a cube did not give a cube with the target beam")
    elif not np.allclose(result.unmasked_data[:].value, expected[np.newaxis]):
        problems.append("the convolved cube does not match the analytic target-beam kernel")

    channel = convolve_uv.convolve_uv(cube[0], TARGET_BEAM)
    if not isinstance(channel, Projection) or channel.beam != TARGET_BEAM:
        problems.append(
            "convolving a single channel did not give a projection with the target beam"
        )
    elif not np.allclose(channel.value, expected):
        problems.append("the convolved channel does not match the analytic target-beam kernel")

    if not problems:
        print(f"convolve_uv {convolve_uv.__version__} from {location.parent}: ok")
    return problems


def main() -> int:
    """Run the checks for the version given on the command line."""
    if len(sys.argv) != 2:
        print("usage: python -I tools/check_install.py EXPECTED_VERSION")
        return 2
    problems = check(sys.argv[1])
    for problem in problems:
        print(f"ERROR: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
