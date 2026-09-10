# convolve_uv

[![PyPI](https://img.shields.io/pypi/v/convolve-uv.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/convolve-uv/)
[![Python](https://img.shields.io/pypi/pyversions/convolve-uv.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/convolve-uv/)
[![build](https://img.shields.io/github/actions/workflow/status/thomaswilliamsastro/convolve_uv/build.yml?branch=main&style=flat-square)](https://github.com/thomaswilliamsastro/convolve_uv/actions/workflows/build.yml)
[![tests](https://img.shields.io/github/actions/workflow/status/thomaswilliamsastro/convolve_uv/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/thomaswilliamsastro/convolve_uv/actions/workflows/tests.yml)
[![readthedocs](https://readthedocs.org/projects/convolve-uv/badge/?version=latest&style=flat-square)](https://convolve-uv.readthedocs.io/en/latest)
[![codecov](https://img.shields.io/codecov/c/gh/thomaswilliamsastro/convolve_uv?style=flat-square)](https://codecov.io/gh/thomaswilliamsastro/convolve_uv)
[![License](https://img.shields.io/badge/license-GNUv3-blue.svg?label=License&style=flat-square)](LICENSE)

`convolve-uv` is a small Python package that allows for image convolution without loss of resolution.

## Installation

```bash
pip install convolve-uv
```

## Usage

```python
import astropy.units as u
from convolve_uv import convolve_uv
from radio_beam import Beam
from spectral_cube import SpectralCube

cube = SpectralCube.read("my_cube.fits")
target_beam = Beam(major=0.86 * u.arcsec, minor=0.86 * u.arcsec, pa=0 * u.deg)

cube_conv = convolve_uv(image=cube, 
                        target_beam=target_beam,
                        )
```

For more details, read the [documentation](https://convolve-uv.readthedocs.io/en/latest/).
