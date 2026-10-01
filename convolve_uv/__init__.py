"""Perform Gaussian convolution directly in the uv-plane."""

from importlib.metadata import PackageNotFoundError, version

from ._convolve import LargeCubeMemoryWarning, convolve_uv

try:
    __version__ = version("convolve_uv")
except PackageNotFoundError:
    # Running from a source tree that has not been installed
    __version__ = "dev"

__all__ = [
    "LargeCubeMemoryWarning",
    "convolve_uv",
]
