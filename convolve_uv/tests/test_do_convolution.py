"""Tests of do_convolution on a single image."""

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube

from .._numerics import (
    do_convolution,
)
from .helpers import (
    _create_test_cube,
    _create_test_varying_resolution_cube,
)


class TestDoConvolution:
    @pytest.mark.parametrize("cube_type", ["spectral", "varying_resolution", "one_channel"])
    def test_do_convolution_rejects_a_cube_with_a_clear_error(self, cube_type):
        """A cube is rejected with a TypeError that says what to pass instead."""
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        if cube_type == "varying_resolution":
            cube = _create_test_varying_resolution_cube(x_size=21, y_size=21, vel_size=2)
        else:
            cube = _create_test_cube(
                x_size=21, y_size=21, vel_size=1 if cube_type == "one_channel" else 2
            )

        with pytest.raises(TypeError, match=r"single 2D image.*3 dimensions.*use convolve_uv"):
            do_convolution(cube, target_beam)

    def test_do_convolution_without_mask(self):
        """Test the convolution helper when a projection has no mask."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]

        class UnmaskedSlice:
            def __init__(self, wrapped_slice):
                self._wrapped_slice = wrapped_slice

            @property
            def mask(self):
                return None

            def __getattr__(self, name):
                return getattr(self._wrapped_slice, name)

        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = do_convolution(UnmaskedSlice(image_slice), target_beam)

        assert np.all(np.isfinite(result))

    def test_do_convolution_with_no_valid_pixels(self):
        """Test convolution returns NaNs when every pixel is masked."""
        base_cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube = base_cube.with_mask(np.zeros(base_cube.shape, dtype=bool))
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = do_convolution(cube[0], target_beam, nan_treatment="fill")

        assert np.all(np.isnan(result))

    def test_do_convolution_with_all_nan_data_and_fill_treatment(self):
        """Test filling an all-NaN image produces the requested fill value."""
        base_cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = np.full(base_cube.shape, np.nan, dtype=np.float32)
        cube = SpectralCube(
            data=data,
            wcs=base_cube.wcs,
            beam=base_cube.beam,
            allow_huge_operations=True,
        )
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = do_convolution(
            cube[0],
            target_beam,
            nan_treatment="fill",
            fill_value=-2.5,
        )

        assert np.allclose(result, -2.5)
