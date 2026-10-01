"""Tests of the kinds of input that convolve_uv and do_convolution accept."""

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube

from .. import convolve_uv
from .helpers import (
    TEST_RESOLUTIONS,
    _create_test_cube,
    _create_test_varying_resolution_cube,
    _get_common_beam,
)


class TestInputs:
    """Slices, cubes, masks and data types."""

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_slice(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a cube slice to a round beam.

        Args:
            common_beam_resolution (u.Quantity | Beam | None): The resolution of the common
                beam to convolve to. Defaults to None, which will calculate a common
                beam from the input cube
        """
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)

        if common_beam_resolution is None:
            common_beam = _get_common_beam(cube.beam)
        elif isinstance(common_beam_resolution, Beam):
            common_beam = common_beam_resolution
        else:
            common_beam = Beam(
                major=common_beam_resolution,
                minor=common_beam_resolution,
                pa=0 * u.deg,
            )

        # Pull a slice from the cube
        cube = cube[0]

        # Do the convolution
        cube_conv = convolve_uv(
            image=cube,
            target_beam=common_beam,
        )
        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[1], y_size=cube.shape[0]
        ).array

        assert cube_conv.beam == common_beam, "Convolved cube beam does not match target beam"
        assert np.allclose(res, analytic_kernel), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_varying_resolution_spectral_cube(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a varying resolution cube to a common beam.

        Args:
            common_beam_resolution (u.Quantity | Beam | None): The resolution of the common
                beam to convolve to. Defaults to None, which will calculate a common
                beam from the input cube
        """
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_varying_resolution_cube(pix_scale=pix_scale)

        if common_beam_resolution is None:
            common_beam = cube.beams.common_beam()
            common_beam = _get_common_beam(common_beam)
        elif isinstance(common_beam_resolution, Beam):
            common_beam = common_beam_resolution
        else:
            common_beam = Beam(
                major=common_beam_resolution,
                minor=common_beam_resolution,
                pa=0 * u.deg,
            )

        # Do the convolution
        cube_conv = convolve_uv(
            image=cube,
            target_beam=common_beam,
        )
        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[2], y_size=cube.shape[1]
        ).array

        assert cube_conv.beam == common_beam, "Convolved cube beam does not match target beam"
        assert np.allclose(res, analytic_kernel), "Convolved kernel does not match analytic kernel"

    def test_convolve_cube_without_mask(self):
        """Test convolution when the input cube has no mask."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube.mask = None
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        cube_conv = convolve_uv(image=cube, target_beam=target_beam)

        assert cube_conv.beam == target_beam
        assert np.all(np.isfinite(cube_conv.unmasked_data[:].value))

    def test_convolve_masked_data(self):
        """Test masked pixels are excluded from convolution weights."""
        base_cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = np.ones(base_cube.shape, dtype=np.float32)
        cube = SpectralCube(
            data=data,
            wcs=base_cube.wcs,
            beam=base_cube.beam,
            fill_value=1e6,
            allow_huge_operations=True,
        )
        mask = np.ones(cube.shape, dtype=bool)
        mask[:, :, 11:] = False
        cube = cube.with_mask(mask)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        cube_conv = convolve_uv(image=cube, target_beam=target_beam)
        res = cube_conv.unmasked_data[:].value

        assert np.allclose(res[:, :, :11], 1.0, atol=1e-5)
        assert np.array_equal(cube_conv.mask.include(), mask)

    @pytest.mark.parametrize("data_dtype", [np.int16, np.bool_])
    def test_convolve_preserves_input_dtype(self, data_dtype: np.dtype):
        """Test convolution preserves the dtype exposed by the cube slice."""
        base_cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = np.zeros(base_cube.shape, dtype=data_dtype)
        data[0, 10, 10] = 1
        cube = SpectralCube(
            data=data,
            wcs=base_cube.wcs,
            beam=base_cube.beam,
            allow_huge_operations=True,
        )
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        expected_dtype = cube[0].dtype

        cube_conv = convolve_uv(image=cube, target_beam=target_beam)
        res = cube_conv.unmasked_data[:].value

        assert res.dtype == np.dtype(expected_dtype)

    @pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
    def test_convolve_slice_different_dtype(
        self,
        data_dtype: np.dtype,
    ):
        """Test convolving a cube slice with a different dtype from default.

        Args:
            data_dtype (np.dtype): dtype to use
        """
        pix_scale = 0.1 * u.arcsec
        common_beam = Beam(major=1 * u.arcsec, minor=1 * u.arcsec, pa=0 * u.deg)
        cube = _create_test_cube(
            pix_scale=pix_scale,
            data_dtype=data_dtype,
        )

        # Pull a slice from the cube
        cube = cube[0]

        # Do the convolution
        cube_conv = convolve_uv(
            image=cube,
            target_beam=common_beam,
        )
        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[1], y_size=cube.shape[0]
        ).array

        assert cube_conv.beam == common_beam, "Convolved cube beam does not match target beam"
        assert np.allclose(res, analytic_kernel), "Convolved kernel does not match analytic kernel"
