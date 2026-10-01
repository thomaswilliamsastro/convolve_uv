"""Tests of the checks on the arguments of convolve_uv and do_convolution."""

import sys
import warnings

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube, cube_utils

from .. import LargeCubeMemoryWarning, _convolve, convolve_uv
from .._numerics import (
    do_convolution,
)
from .helpers import (
    _create_test_cube,
    _FakeTerminal,
    _get_common_beam,
    _run_convolution,
)


class TestArgumentValidation:
    """Arguments that are rejected, ignored or allowed."""

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


# Values that are not valid for fill_value, preserve_nan or show_progress, and their type names
BAD_FILL_VALUES: list[tuple[object, str]] = [
    ("abc", "str"),
    (None, "NoneType"),
    (1 + 2j, "complex"),
    ([1.0], "list"),
    (np.array([1.0, 2.0]), "ndarray"),
    (5 * u.K, "Quantity"),
]
BAD_PRESERVE_NAN: list[tuple[object, str]] = [
    ("False", "str"),
    (None, "NoneType"),
    (1, "int"),
    ([], "list"),
]


class TestCubeArgumentValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"boundary": "reflect"}, ValueError, "boundary must be"),
            ({"pad_sigma": -1.0}, ValueError, "pad_sigma must be"),
            ({"nan_treatment": "drop"}, ValueError, "nan_treatment must be"),
            ({"target_beam": 1.5 * u.arcsec}, TypeError, "must be a Beam object"),
            *[
                (
                    {"nan_treatment": "fill", "fill_value": value},
                    TypeError,
                    f"fill_value must be a real number, not {name}",
                )
                for value, name in BAD_FILL_VALUES
            ],
            *[
                (
                    {"preserve_nan": value},
                    TypeError,
                    f"preserve_nan must be True or False, not {name}",
                )
                for value, name in BAD_PRESERVE_NAN
            ],
            *[
                (
                    {"show_progress": value},
                    TypeError,
                    f"show_progress must be True or False, not {name}",
                )
                for value, name in BAD_PRESERVE_NAN
            ],
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

    @pytest.mark.parametrize("operation", ["convolve_uv", "do_convolution"])
    def test_both_entry_points_reject_a_bad_fill_value_and_preserve_nan(self, operation):
        """The checks are shared, so convolve_uv and do_convolution give the same errors."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        image = cube if operation == "convolve_uv" else cube[0]

        with pytest.raises(TypeError, match="fill_value must be a real number, not str"):
            _run_convolution(
                operation, image, target_beam=target_beam, nan_treatment="fill", fill_value="0"
            )
        with pytest.raises(TypeError, match="preserve_nan must be True or False, not str"):
            _run_convolution(operation, image, target_beam=target_beam, preserve_nan="False")

    @pytest.mark.parametrize("fill_value", [0, 2, True, 1.5, np.float32(1.5), np.int64(3)])
    def test_any_real_number_is_a_valid_fill_value(self, fill_value):
        """Integers, floats, bools and numpy scalars are all real numbers."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = convolve_uv(cube, target_beam, nan_treatment="fill", fill_value=fill_value)

        assert np.isfinite(result.unmasked_data[:].value).all()

    @pytest.mark.parametrize("preserve_nan", [np.True_, np.False_])
    def test_numpy_bools_are_valid_for_preserve_nan(self, preserve_nan):
        """A numpy bool is still a bool, for example one taken from an array."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = convolve_uv(cube, target_beam, preserve_nan=preserve_nan)

        assert result.shape == cube.shape

    @pytest.mark.parametrize("fill_value", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_fill_value_leaves_the_pixels_out(self, fill_value):
        """A non-finite fill_value (nan or inf) is allowed: it leaves the NaN pixels out.

        With a real fill_value the NaN counts as that value, so a NaN in an image of ones pulls
        the result there below 1. With a non-finite one the pixel is left out of the average, so
        an image of ones stays exactly one.
        """
        data = np.ones((1, 41, 41))
        data[0, 20, 20] = np.nan
        template = _create_test_cube(x_size=41, y_size=41, vel_size=1, pix_scale=1 * u.arcsec)
        cube = SpectralCube(
            data=data, wcs=template.wcs, beam=template.beam, allow_huge_operations=True
        )
        target_beam = Beam(major=8 * u.arcsec, minor=8 * u.arcsec, pa=0 * u.deg)

        left_out = convolve_uv(cube, target_beam, nan_treatment="fill", fill_value=fill_value)
        counted_as_zero = convolve_uv(cube, target_beam, nan_treatment="fill", fill_value=0.0)

        np.testing.assert_allclose(left_out.unmasked_data[:].value, 1.0, atol=1e-6)
        assert counted_as_zero.unmasked_data[:].value[0, 20, 20] < 0.99

    @pytest.mark.parametrize("fill_value", [5.0, np.nan, "abc", None])
    def test_fill_value_is_ignored_and_unchecked_when_interpolating(self, fill_value):
        """Only nan_treatment='fill' uses fill_value, so anything is accepted otherwise.

        The default boundary pads the image, and the padding used to be filled with fill_value,
        so this also shows that the result does not depend on it.
        """
        data = np.ones((1, 41, 41))
        data[0, 20, 20] = np.nan
        template = _create_test_cube(x_size=41, y_size=41, vel_size=1, pix_scale=1 * u.arcsec)
        cube = SpectralCube(
            data=data, wcs=template.wcs, beam=template.beam, allow_huge_operations=True
        )
        target_beam = Beam(major=8 * u.arcsec, minor=8 * u.arcsec, pa=0 * u.deg)

        reference = convolve_uv(cube, target_beam).unmasked_data[:].value
        result = convolve_uv(cube, target_beam, fill_value=fill_value).unmasked_data[:].value

        np.testing.assert_array_equal(result, reference)

    @pytest.mark.parametrize("show_progress", ["False", "no", 0, None])
    def test_show_progress_is_checked_for_a_projection_too(self, show_progress):
        """It is checked even though a projection has no bar, so a mistake is not hidden."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(TypeError, match="show_progress must be True or False"):
            convolve_uv(cube[0], target_beam, show_progress=show_progress)

    def test_a_string_no_longer_draws_the_progress_bar(self, monkeypatch):
        """show_progress="False" used to be truthy and draw the bar, as True does."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        # The wrong type is the point, so mypy's complaint about it is silenced on this call
        with pytest.raises(TypeError, match="show_progress must be True or False, not str"):
            convolve_uv(
                _create_test_cube(vel_size=3),
                target_beam,
                show_progress="False",  # type: ignore[arg-type]
            )

        assert terminal.getvalue() == ""
