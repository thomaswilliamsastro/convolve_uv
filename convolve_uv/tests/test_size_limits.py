"""Tests of the limit on the size of the arrays that are allocated."""

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam

from .. import convolve_uv
from .._numerics import (
    do_convolution,
)
from .helpers import (
    BOUNDARY_KEYWORDS,
    _create_test_cube,
)


class TestArraySizeLimits:
    """Absurdly large padding or kernel sizes are rejected before allocating."""

    WIDE_BEAM = Beam(major=1e5 * u.arcsec, minor=1e5 * u.arcsec, pa=0 * u.deg)
    ROUND_BEAM = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

    @staticmethod
    def _convolve(operation: str, cube=None, **kwargs):
        if cube is None:
            cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        if operation == "convolve_uv":
            return convolve_uv(image=cube, **kwargs)
        return do_convolution(image_slice=cube[0], **kwargs)

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    @pytest.mark.parametrize("pad_sigma", [1e6, 1e308])
    def test_huge_pad_sigma_is_rejected(self, operation: str, pad_sigma: float):
        """A huge (even overflowing) pad_sigma raises a clear error, not a MemoryError."""
        with pytest.raises(ValueError, match="The padded image would need about"):
            self._convolve(
                operation,
                target_beam=self.ROUND_BEAM,
                boundary="fill",
                nan_treatment="fill",
                pad_sigma=pad_sigma,
            )

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    def test_wide_beam_padding_is_rejected(self, operation: str):
        """A beam far wider than the pixel scale makes the padded image too large."""
        with pytest.raises(ValueError, match="The padded image would need about"):
            self._convolve(
                operation,
                target_beam=self.WIDE_BEAM,
                boundary="fill",
                nan_treatment="fill",
            )

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    def test_wide_beam_nan_kernel_is_clipped_to_the_image(self, operation: str, boundary: str):
        """A beam far wider than the pixel scale does not make the NaN kernel too large.

        The kernel is clipped to the image, which is all that it can reach.
        """
        result = self._convolve(
            operation,
            target_beam=self.WIDE_BEAM,
            boundary=boundary,
            nan_treatment="interpolate",
            pad_sigma=0,
        )
        result_data = result.unmasked_data[:].value if operation == "convolve_uv" else result

        assert np.all(np.isfinite(result_data))

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    def test_wide_beam_nan_kernel_for_a_long_image_is_rejected(self, operation: str):
        """The kernel can only be as large as the image is long, and that can be too large.

        This does not depend on pad_sigma or the boundary.
        """
        cube = _create_test_cube(x_size=2**14 + 1, y_size=1, vel_size=1)
        with pytest.raises(ValueError, match="The NaN interpolation kernel would need about"):
            self._convolve(
                operation,
                cube=cube,
                target_beam=self.WIDE_BEAM,
                boundary="wrap",
                nan_treatment="interpolate",
            )

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    def test_wide_beam_without_padding_or_kernel_is_allowed(self, operation: str):
        """The limit only applies to arrays that are actually allocated."""
        result = self._convolve(
            operation,
            target_beam=self.WIDE_BEAM,
            boundary="wrap",
            nan_treatment="fill",
        )
        result_data = result.unmasked_data[:].value if operation == "convolve_uv" else result

        assert np.all(np.isfinite(result_data))
