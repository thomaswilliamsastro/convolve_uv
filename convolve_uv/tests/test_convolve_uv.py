"""Tests of the public convolve_uv function, and of the checks shared with do_convolution."""

import io
import itertools
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube, cube_utils
from spectral_cube.masks import BooleanArrayMask

from .. import LargeCubeMemoryWarning, _convolve, convolve_uv
from .._numerics import do_convolution
from .helpers import (
    BOUNDARY_KEYWORDS,
    _create_anisotropic_wcs_cube,
    _create_test_cube,
    _create_test_varying_resolution_cube,
    _get_common_beam,
)

TEST_RESOLUTIONS = [
    None,
    1 * u.arcsec,
    1.5 * u.arcsec,
    Beam(major=1.5 * u.arcsec, minor=1.3 * u.arcsec, pa=45 * u.deg),
]
NAN_TREATMENT_KEYWORDS = [
    "interpolate",
    "fill",
]


def _run_convolution(operation: str, image, **kwargs):
    """Run either public convolution entry point on an image.

    Args:
        operation (str): ``"convolve_uv"`` or ``"do_convolution"``.
        image: A cube (for ``convolve_uv``) or a projection (for ``do_convolution``).
        **kwargs: Passed on to the chosen function.
    """
    if operation == "convolve_uv":
        return convolve_uv(image=image, **kwargs)
    return do_convolution(image_slice=image, **kwargs)


class TestConvolveUV:
    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    @pytest.mark.parametrize("pad_sigma", [-1.0, np.nan, np.inf, -np.inf, None])
    def test_invalid_pad_sigma(
        self,
        operation: str,
        pad_sigma: float,
    ):
        """Both public convolution entry points reject invalid pad_sigma values."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        image = cube if operation == "convolve_uv" else cube[0]

        with pytest.raises(ValueError, match="pad_sigma must be a finite, non-negative number"):
            _run_convolution(operation, image, target_beam=target_beam, pad_sigma=pad_sigma)

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    def test_zero_pad_sigma_is_allowed(self, operation: str):
        """A zero-width boundary pad is valid for both public entry points."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        image = cube if operation == "convolve_uv" else cube[0]

        if operation == "convolve_uv":
            result = convolve_uv(
                image=image,
                target_beam=target_beam,
                pad_sigma=0,
                nan_treatment="fill",
            )
            result_data = result.unmasked_data[:].value
        else:
            result = do_convolution(
                image_slice=image,
                target_beam=target_beam,
                pad_sigma=0,
                nan_treatment="fill",
            )
            result_data = result

        assert result.shape == image.shape
        assert np.all(np.isfinite(result_data))

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    @pytest.mark.parametrize("pad_sigma", [-1.0, np.nan, np.inf, -np.inf, None])
    def test_invalid_pad_sigma_is_ignored_for_wrap(
        self,
        operation: str,
        pad_sigma: float,
    ):
        """Padding values are not validated when wrapping is selected."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        image = cube if operation == "convolve_uv" else cube[0]

        if operation == "convolve_uv":
            result = convolve_uv(
                image=image,
                target_beam=target_beam,
                boundary="wrap",
                pad_sigma=pad_sigma,
                nan_treatment="fill",
            )
            result_data = result.unmasked_data[:].value
        else:
            result_data = do_convolution(
                image_slice=image,
                target_beam=target_beam,
                boundary="wrap",
                pad_sigma=pad_sigma,
                nan_treatment="fill",
            )

        assert np.all(np.isfinite(result_data))

    def test_non_valid_boundary(self):
        """Test passing a non-valid boundary keyword."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        with pytest.raises(ValueError, match="boundary must be"):
            convolve_uv(
                image=cube,
                target_beam=common_beam,
                boundary="this_should_fail",
            )
        with pytest.raises(ValueError, match="boundary must be"):
            do_convolution(
                image_slice=cube[0],
                target_beam=common_beam,
                boundary="this_should_fail",
            )

    def test_singular_wcs(self):
        """Test passing a singular WCS."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)

        # Manually modify to a totally broken WCS
        cube.wcs.wcs.pc = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        common_beam = _get_common_beam(cube.beam)

        with pytest.raises(ValueError, match="singular pixel-scale matrix"):
            convolve_uv(
                image=cube,
                target_beam=common_beam,
            )

    def test_non_finite_wcs(self):
        """Test a non-finite celestial WCS is rejected with a contextual error."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube.wcs.wcs.pc = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match="celestial WCS is invalid or singular"):
            convolve_uv(image=cube, target_beam=target_beam)

    def test_non_valid_nan_treatment(self):
        """Test passing a non-valid nan_treatment keyword."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        with pytest.raises(ValueError, match="nan_treatment must be"):
            convolve_uv(
                image=cube,
                target_beam=common_beam,
                nan_treatment="this_should_fail",
            )

    def test_non_valid_cube_beam(self):
        """Test passing a cube without a beam."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale, beam=None)
        common_beam = Beam(major=1 * u.arcsec, minor=1 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(AttributeError, match="image_slice must have a valid beam"):
            convolve_uv(
                image=cube,
                target_beam=common_beam,
            )

    def test_non_valid_target_beam(self):
        """Test passing a non-valid target beam."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)

        with pytest.raises(TypeError, match="Input beam must be a Beam object"):
            convolve_uv(
                image=cube,
                target_beam="this_should_fail",
            )

    def test_too_small_target_beam(self):
        """Test passing a too-small target beam."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        target_beam = Beam(0.1 * u.arcsec, 0.1 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match="target beam is smaller than the input beam"):
            convolve_uv(
                image=cube,
                target_beam=target_beam,
            )

    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    def test_boundary_keywords(
        self,
        boundary: str,
    ):
        """Test passing boundary keywords.

        Args:
            boundary (str): The boundary keyword to test.
        """
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        cube_conv = convolve_uv(image=cube, target_beam=common_beam, boundary=boundary)

        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[2], y_size=cube.shape[1]
        ).array

        assert cube_conv.beam == common_beam, "Convolved cube beam does not match target beam"
        assert np.allclose(res, analytic_kernel[np.newaxis, ...]), (
            "Convolved kernel does not match analytic kernel"
        )

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

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_cube(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a simple cube to a round beam.

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
        assert np.allclose(res, analytic_kernel[np.newaxis, ...]), (
            "Convolved kernel does not match analytic kernel"
        )

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_cube_jy_beam(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a simple cube in Jy/beam to a round beam.

        Args:
            common_beam_resolution (u.Quantity | Beam | None): The resolution of the common
                beam to convolve to. Defaults to None, which will calculate a common
                beam from the input cube
        """
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale, unit=u.Jy / u.beam)

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

        # This will have rescaled the data a bit
        beam_ratio_factor = (common_beam.sr / cube.beam.sr).value
        analytic_kernel *= beam_ratio_factor

        assert cube_conv.beam == common_beam, "Convolved cube beam does not match target beam"
        assert np.allclose(res, analytic_kernel[np.newaxis, ...]), (
            "Convolved kernel does not match analytic kernel"
        )

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

    @pytest.mark.parametrize("flag", [True, False])
    def test_convolve_cube_keeps_allow_huge_operations(self, flag: bool):
        """The input's allow_huge_operations is unchanged, and the result has the same one."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = flag
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = convolve_uv(image=cube, target_beam=target_beam)

        assert cube.allow_huge_operations is flag
        assert result.allow_huge_operations is flag

    @pytest.mark.parametrize(
        ("target_arcsec", "error"),
        [(1.5, None), (0.1, ValueError)],
        ids=["succeeds", "raises part-way through"],
    )
    def test_convolve_cube_never_writes_allow_huge_operations(
        self, target_arcsec: float, error: type[Exception] | None
    ):
        """The input is never written to, not even temporarily, whether or not the call succeeds."""
        writes = []

        # spectral-cube ships no type information, so its cube class is Any to mypy
        class RecordingCube(SpectralCube):  # type: ignore[misc]
            def __setattr__(self, name, value):
                if getattr(self, "_recording", False):
                    writes.append(name)
                super().__setattr__(name, value)

        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        cube.__class__ = RecordingCube
        cube._recording = True
        cube._recorder_check = True
        target_beam = Beam(
            major=target_arcsec * u.arcsec, minor=target_arcsec * u.arcsec, pa=0 * u.deg
        )
        expectation = (
            pytest.raises(error, match="smaller than the input beam") if error else nullcontext()
        )

        with expectation:
            convolve_uv(image=cube, target_beam=target_beam)

        assert "_recorder_check" in writes
        assert "allow_huge_operations" not in writes

    def test_concurrent_convolutions_of_one_cube_do_not_interfere(self, monkeypatch):
        """Two overlapping calls on one cube leave its allow_huge_operations alone.

        The second call starts once the first is part-way through and finishes last, the
        order in which temporarily overriding the flag used to leave it stuck on True.
        """
        original = do_convolution
        callers = itertools.count()
        lock = threading.Lock()
        first_inside = threading.Event()
        first_done = threading.Event()
        both_inside = threading.Barrier(2, timeout=30)

        def hold(*args, **kwargs):
            with lock:
                position = next(callers)
            if position == 0:
                first_inside.set()
            both_inside.wait()
            if position == 1:
                first_done.wait(30)
            return original(*args, **kwargs)

        monkeypatch.setattr(_convolve, "do_convolution", hold)
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(convolve_uv, image=cube, target_beam=target_beam)
            first.add_done_callback(lambda _: first_done.set())
            assert first_inside.wait(30)
            second = pool.submit(convolve_uv, image=cube, target_beam=target_beam)
            first_result, second_result = first.result(60), second.result(60)

        assert cube.allow_huge_operations is False
        assert np.array_equal(
            first_result.unmasked_data[:].value, second_result.unmasked_data[:].value
        )

    def test_convolve_projection_does_not_leak_allow_huge_operations(self):
        """Test convolve_uv doesn't leave a new attribute on a Projection without one."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        assert not hasattr(image_slice, "allow_huge_operations")
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        convolve_uv(image=image_slice, target_beam=target_beam)

        assert not hasattr(image_slice, "allow_huge_operations")

    def test_convolve_projection_keeps_allow_huge_operations_on_error(self):
        """Test convolve_uv keeps a Projection's allow_huge_operations on error."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        image_slice.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match="boundary must be"):
            convolve_uv(
                image=image_slice,
                target_beam=target_beam,
                boundary="invalid",
            )

        assert image_slice.allow_huge_operations is False

    def test_convolve_cube_below_huge_threshold_is_quiet(self):
        """Test convolve_uv does not warn for a cube below the huge-operation threshold."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        assert cube.size < cube_utils.MEMORY_THRESHOLD
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("error", LargeCubeMemoryWarning)
            convolve_uv(image=cube, target_beam=target_beam)

    def test_convolve_cube_at_huge_threshold_warns(self, monkeypatch):
        """Test convolve_uv warns once a cube's size reaches the huge-operation threshold.

        The threshold itself (``spectral_cube.cube_utils.MEMORY_THRESHOLD``) is lowered so the
        test can exercise the boundary without allocating a genuinely huge cube.
        """
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", cube.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.warns(LargeCubeMemoryWarning, match=str(cube.size)):
            convolve_uv(image=cube, target_beam=target_beam)

    def test_convolve_cube_huge_warning_does_not_mutate_allow_huge_operations(self, monkeypatch):
        """Test the large-cube warning leaves the caller's allow_huge_operations untouched."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", cube.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.warns(LargeCubeMemoryWarning):
            convolve_uv(image=cube, target_beam=target_beam)

        assert cube.allow_huge_operations is False

    def test_convolve_projection_never_warns_about_huge_operations(self, monkeypatch):
        """Test convolve_uv never emits the large-cube warning for a bare Projection.

        Projections don't carry an ``allow_huge_operations`` attribute in spectral-cube, so
        the huge-operation check/warning should never apply to them regardless of size.
        """
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", image_slice.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("error", LargeCubeMemoryWarning)
            convolve_uv(image=image_slice, target_beam=target_beam)

    def test_convolve_uv_nan_interpolation_anisotropic_rotated_wcs(self):
        """Test NaN interpolation works end-to-end with anisotropic/rotated pixels."""
        cube = _create_anisotropic_wcs_cube(x_size=41, y_size=41, vel_size=1)
        data = cube.unmasked_data[:].value.copy()
        data[0, 20, 20] = np.nan
        cube = SpectralCube(
            data=data,
            wcs=cube.wcs,
            beam=cube.beam,
            allow_huge_operations=True,
        )
        target_beam = _get_common_beam(cube.beam)

        cube_conv = convolve_uv(
            image=cube,
            target_beam=target_beam,
            nan_treatment="interpolate",
            preserve_nan=False,
        )
        res = cube_conv.unitless_filled_data[:]

        assert cube_conv.beam == target_beam
        assert np.all(np.isfinite(res))

    def test_convolve_uv_preserves_ordinary_square_pixels(self):
        """Test NaN interpolation is unchanged for ordinary, square pixels."""
        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1, pix_scale=pix_scale)
        data = cube.unmasked_data[:].value.copy()
        data[0, 10, 10] = np.nan
        cube = SpectralCube(
            data=data,
            wcs=cube.wcs,
            beam=cube.beam,
            allow_huge_operations=True,
        )
        common_beam = _get_common_beam(cube.beam)

        cube_conv = convolve_uv(
            image=cube,
            target_beam=common_beam,
            nan_treatment="interpolate",
        )
        res = cube_conv.unitless_filled_data[:]

        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[2], y_size=cube.shape[1]
        ).array

        assert cube_conv.beam == common_beam
        assert np.all(np.isfinite(res))
        assert np.allclose(res, analytic_kernel[np.newaxis, ...], atol=2e-3)


class TestArraySizeLimits:
    """Absurdly large padding or kernel sizes are rejected before allocating."""

    WIDE_BEAM = Beam(major=1e5 * u.arcsec, minor=1e5 * u.arcsec, pa=0 * u.deg)
    ROUND_BEAM = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

    @staticmethod
    def _convolve(operation: str, **kwargs):
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
    def test_wide_beam_nan_kernel_is_rejected(self, operation: str, boundary: str):
        """A beam far wider than the pixel scale makes the NaN kernel too large.

        This does not depend on pad_sigma or the boundary.
        """
        with pytest.raises(ValueError, match="The NaN interpolation kernel would need about"):
            self._convolve(
                operation,
                target_beam=self.WIDE_BEAM,
                boundary=boundary,
                nan_treatment="interpolate",
                pad_sigma=0,
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


class _FakeTerminal(io.StringIO):
    """A writable stand-in for standard output that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


class TestShowProgress:
    # The progress bar only draws when standard output is a terminal, so these tests make
    # sys.stdout look like one, and read back what was written to it.
    target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

    def test_the_bar_is_drawn_on_a_terminal_by_default(self, monkeypatch):
        """With no keyword, a cube convolved on a terminal gets a progress bar."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)

        convolve_uv(_create_test_cube(vel_size=3), self.target_beam)

        assert "100.00%" in terminal.getvalue()

    def test_the_bar_can_be_switched_off(self, monkeypatch):
        """With show_progress=False nothing is written to a terminal."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)

        convolve_uv(_create_test_cube(vel_size=3), self.target_beam, show_progress=False)

        assert terminal.getvalue() == ""

    def test_the_bar_does_not_change_the_result(self):
        """The convolved cube is the same with and without the bar."""
        cube = _create_test_cube(vel_size=3)

        with_bar = convolve_uv(cube, self.target_beam, show_progress=True)
        without_bar = convolve_uv(cube, self.target_beam, show_progress=False)

        np.testing.assert_array_equal(with_bar.unmasked_data[:], without_bar.unmasked_data[:])

    def test_a_cube_can_be_convolved_in_a_worker_thread_on_a_terminal(self, monkeypatch):
        """Without a bar there is no signal handler to install, so any thread can convolve."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)
        cube = _create_test_cube(vel_size=3)

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(convolve_uv, cube, self.target_beam, show_progress=False).result()

        assert result.shape == cube.shape
        assert terminal.getvalue() == ""

    def test_a_projection_ignores_the_keyword(self, monkeypatch):
        """A projection has no bar, so the keyword changes nothing for it."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)
        projection = _create_test_cube(vel_size=2)[0]

        shown = convolve_uv(projection, self.target_beam, show_progress=True)
        hidden = convolve_uv(projection, self.target_beam, show_progress=False)

        np.testing.assert_array_equal(shown.value, hidden.value)
        assert terminal.getvalue() == ""


class TestCubeArgumentValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"boundary": "reflect"}, ValueError, "boundary must be"),
            ({"pad_sigma": -1.0}, ValueError, "pad_sigma must be"),
            ({"nan_treatment": "drop"}, ValueError, "nan_treatment must be"),
            ({"target_beam": 1.5 * u.arcsec}, TypeError, "must be a Beam object"),
        ],
    )
    def test_a_bad_argument_is_rejected_before_the_cube_is_touched(
        self, monkeypatch, kwargs, error, match
    ):
        """A cube with a bad argument fails at once, not on its first channel.

        The size warning is turned into an error and the per-channel convolution is
        replaced by a recorder, so the call only raises the expected error if it is rejected
        before the warning and before any channel is convolved.
        """
        calls = []
        monkeypatch.setattr(_convolve, "do_convolution", lambda *args, **kwargs: calls.append(1))
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", 1)
        arguments = {
            "target_beam": Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg),
            **kwargs,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", LargeCubeMemoryWarning)
            with pytest.raises(error, match=match):
                convolve_uv(_create_test_cube(vel_size=2), **arguments)

        assert calls == []

    def test_pad_sigma_is_not_checked_when_the_boundary_wraps(self):
        """Shared validation still ignores pad_sigma for boundary='wrap', as documented."""
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = convolve_uv(
            _create_test_cube(vel_size=1), target_beam, boundary="wrap", pad_sigma=-1.0
        )

        assert result.shape[0] == 1
