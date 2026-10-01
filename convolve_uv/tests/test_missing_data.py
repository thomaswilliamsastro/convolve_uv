"""Tests of NaNs and masked pixels, and of images that already have the target beam."""

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.masks import BooleanArrayMask

from .. import convolve_uv
from .helpers import (
    NAN_TREATMENT_KEYWORDS,
    _create_test_cube,
    _create_test_varying_resolution_cube,
)


class TestMissingData:
    """What happens to NaN and masked pixels."""

    @pytest.mark.parametrize("nan_treatment", NAN_TREATMENT_KEYWORDS)
    @pytest.mark.parametrize("preserve_nan", [True, False])
    def test_nan_treatment_keywords(
        self,
        nan_treatment: str,
        preserve_nan: bool,
    ):
        """Test NaN handling during a non-identity beam convolution.

        Args:
            nan_treatment (str): The nan_treatment keyword to test.
            preserve_nan (bool): Whether to restore the original NaN positions.
        """
        base_cube = _create_test_cube(x_size=41, y_size=41, vel_size=1)
        data = np.ones(base_cube.shape, dtype=np.float32)
        data[0, 20, 20] = np.nan
        cube = SpectralCube(
            data=data,
            wcs=base_cube.wcs,
            beam=base_cube.beam,
            allow_huge_operations=True,
        )
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        cube_conv = convolve_uv(
            image=cube,
            target_beam=target_beam,
            nan_treatment=nan_treatment,
            fill_value=-100.0,
            preserve_nan=preserve_nan,
        )
        res = cube_conv.unmasked_data[:].value

        assert cube_conv.beam == target_beam
        expected_nan = np.zeros(res.shape, dtype=bool)
        if preserve_nan:
            expected_nan[0, 20, 20] = True
        assert np.array_equal(np.isnan(res), expected_nan)
        if not preserve_nan:
            assert np.all(np.isfinite(res))
            if nan_treatment == "interpolate":
                assert np.isclose(res[0, 20, 20], 1.0, atol=1e-5)
            else:
                assert res[0, 20, 20] < 0.9

    def test_convolve_slice_same_beam(self):
        """Test convolving a cube slice to the same beam."""
        pix_scale = 0.1 * u.arcsec
        common_beam = Beam(major=1 * u.arcsec, minor=1 * u.arcsec, pa=0 * u.deg)
        cube = _create_test_cube(pix_scale=pix_scale, beam=common_beam)

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

    def test_same_beam_preserves_nan(self):
        """Test an identical-beam operation preserves NaN values."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = cube.unmasked_data[:].value.copy()
        data[0, 10, 10] = np.nan
        cube = SpectralCube(
            data=data,
            wcs=cube.wcs,
            beam=cube.beam,
            allow_huge_operations=True,
        )

        cube_conv = convolve_uv(image=cube, target_beam=cube.beam)
        res = cube_conv.unmasked_data[:].value

        assert np.isnan(res[0, 10, 10])
        assert np.allclose(res[0, :10, :], data[0, :10, :], equal_nan=True)

    @pytest.mark.parametrize(
        ("preserve_nan", "expected_nan"),
        [(False, False), (True, True)],
    )
    def test_same_beam_nan_treatment_fill(
        self,
        preserve_nan: bool,
        expected_nan: bool,
    ):
        """Test fill and preserve_nan behavior for identical beams."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = cube.unmasked_data[:].value.copy()
        data[0, 10, 10] = np.nan
        cube = SpectralCube(
            data=data,
            wcs=cube.wcs,
            beam=cube.beam,
            allow_huge_operations=True,
        )

        cube_conv = convolve_uv(
            image=cube,
            target_beam=cube.beam,
            nan_treatment="fill",
            fill_value=-3.0,
            preserve_nan=preserve_nan,
        )
        res = cube_conv.unmasked_data[:].value

        assert np.isnan(res[0, 10, 10]) == expected_nan
        if not expected_nan:
            assert res[0, 10, 10] == -3.0

    @pytest.mark.parametrize(
        ("nan_treatment", "masked_value"),
        [("interpolate", np.nan), ("fill", -3.0)],
    )
    def test_same_beam_masked_pixels_are_treated_as_missing(self, nan_treatment, masked_value):
        """An identical-beam image returns the valid data and treats masked pixels as NaN."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        data = cube.unmasked_data[:].value.copy()
        mask = np.ones(data.shape, dtype=bool)
        mask[0, 5:9, 12:16] = False
        cube = SpectralCube(
            data=data,
            wcs=cube.wcs,
            beam=cube.beam,
            mask=BooleanArrayMask(mask, cube.wcs),
            allow_huge_operations=True,
        )

        cube_conv = convolve_uv(
            image=cube, target_beam=cube.beam, nan_treatment=nan_treatment, fill_value=-3.0
        )
        res = cube_conv.unmasked_data[:].value

        np.testing.assert_array_equal(res[mask], data[mask])
        np.testing.assert_array_equal(res[~mask], np.full(np.count_nonzero(~mask), masked_value))
        np.testing.assert_array_equal(cube_conv.mask.include(), mask)

    @pytest.mark.parametrize(
        ("nan_treatment", "expected_nan"),
        [("interpolate", [False, False, False, True]), ("fill", [False] * 4)],
    )
    def test_nans_are_only_interpolated_in_channels_that_are_convolved(
        self, nan_treatment, expected_nan
    ):
        """With interpolation, the channel already at the target beam keeps its NaN.

        This is the documented behaviour when convolving a varying-resolution cube to its
        common beam: that beam is the widest channel's, which needs no convolution.
        """
        cube = _create_test_varying_resolution_cube(x_size=31, y_size=31, vel_size=4)
        data = np.array(cube.unmasked_data[:])
        data[:, 10, 10] = np.nan
        cube = VaryingResolutionSpectralCube(
            data=data, wcs=cube.wcs, beams=cube.beams, allow_huge_operations=True
        )
        target_beam = cube.beams.common_beam()

        res = convolve_uv(cube, target_beam, nan_treatment=nan_treatment).unmasked_data[:].value

        assert [bool(b == target_beam) for b in cube.beams] == [False, False, False, True]
        assert list(np.isnan(res[:, 10, 10])) == expected_nan
