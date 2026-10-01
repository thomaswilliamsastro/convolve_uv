"""Tests of the covariance of the beams, the kernel and the pixel grid."""

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam

from .._numerics import (
    FWHM_TO_SIGMA,
    beam_covariance_en,
    do_convolution,
    kernel_covariance_pixels,
)
from .helpers import (
    DEFAULT_BEAM,
    _create_anisotropic_wcs_cube,
    _create_test_cube,
    _get_common_beam,
)


class TestCovariance:
    @pytest.mark.parametrize(
        ("pixel_scale_matrix", "message"),
        [
            (np.ones((1, 1)), "finite 2x2 values"),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite 2x2 values"),
        ],
    )
    def test_invalid_pixel_scale_matrix(
        self,
        pixel_scale_matrix: np.ndarray,
        message: str,
    ):
        """Malformed pixel-scale matrices raise a contextual error."""

        class TestWCS:
            def __init__(self, matrix):
                self.celestial = self
                self.pixel_scale_matrix = matrix

        class TestSlice:
            def __init__(self):
                self.beam = DEFAULT_BEAM
                self.wcs = TestWCS(pixel_scale_matrix)

        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match=message):
            kernel_covariance_pixels(TestSlice(), target_beam)

    def test_wcs_matrix_error_is_contextual(self):
        """WCS conversion errors are wrapped with pixel-scale context."""

        class TestWCS:
            def __init__(self):
                self.celestial = self

            @property
            def pixel_scale_matrix(self):
                raise ValueError("invalid WCS")

        class TestSlice:
            def __init__(self):
                self.beam = DEFAULT_BEAM
                self.wcs = TestWCS()

        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(
            ValueError, match="Unable to compute the celestial WCS pixel-scale matrix"
        ):
            kernel_covariance_pixels(TestSlice(), target_beam)

    def test_wcs_matrix_inverse_error_is_contextual(self, monkeypatch):
        """A numerical WCS inversion failure is reported as a singular matrix."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        def fail_inverse(_matrix):
            raise np.linalg.LinAlgError("test failure")

        monkeypatch.setattr(np.linalg, "inv", fail_inverse)

        with pytest.raises(ValueError, match="singular pixel-scale matrix"):
            kernel_covariance_pixels(cube[0], target_beam)

    def test_non_finite_wcs_inverse_is_contextual(self, monkeypatch):
        """A non-finite WCS inverse is rejected before covariance math."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)
        monkeypatch.setattr(np.linalg, "inv", lambda _matrix: np.array([[np.inf, 0.0], [0.0, 1.0]]))

        with pytest.raises(ValueError, match="pixel-scale matrix has a non-finite inverse"):
            kernel_covariance_pixels(cube[0], target_beam)

    def test_wcs_covariance_transform_error_is_contextual(self):
        """Overflow while transforming a valid covariance reports WCS context."""

        class TestWCS:
            def __init__(self):
                self.celestial = self
                self.pixel_scale_matrix = np.eye(2) * 1e-160

        class TestSlice:
            def __init__(self):
                self.beam = DEFAULT_BEAM
                self.wcs = TestWCS()

        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with (
            np.errstate(over="raise"),
            pytest.raises(
                ValueError,
                match="Pixel-space kernel covariance must be a finite 2x2 covariance matrix",
            ),
        ):
            kernel_covariance_pixels(TestSlice(), target_beam)

    def test_beam_covariance_invalid_beam_is_contextual(self):
        """Invalid beam inputs raise a contextual error."""
        with pytest.raises(ValueError, match="Beam covariance requires finite angular"):
            beam_covariance_en(None)

    @pytest.mark.parametrize(
        ("pa_deg", "expected"),
        [
            (0.0, [[1.0, 0.0], [0.0, 4.0]]),
            (90.0, [[4.0, 0.0], [0.0, 1.0]]),
            (180.0, [[1.0, 0.0], [0.0, 4.0]]),
            (45.0, [[2.5, 1.5], [1.5, 2.5]]),
            (-45.0, [[2.5, -1.5], [-1.5, 2.5]]),
        ],
    )
    def test_beam_covariance_follows_the_position_angle(
        self, pa_deg: float, expected: list[list[float]]
    ):
        """The major axis lies at the position angle, measured from north towards east.

        The beam has sigma 2 and 1 in whatever units its major and minor axes are given in,
        so the covariance is 4 along the major axis and 1 along the minor axis.
        """
        sigma_to_fwhm = 1.0 / FWHM_TO_SIGMA
        beam = Beam(
            major=2.0 * sigma_to_fwhm * u.deg, minor=1.0 * sigma_to_fwhm * u.deg, pa=pa_deg * u.deg
        )

        assert np.allclose(beam_covariance_en(beam), expected, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize(
        ("unit", "per_degree"),
        [
            (u.deg, 1.0),
            (u.rad, np.pi / 180.0),
            (u.arcmin, 60.0),
            (u.arcsec, 3600.0),
            (u.cycle, 1.0 / 360.0),
        ],
        ids=str,
    )
    def test_beam_covariance_position_angle_in_any_angular_unit(self, unit, per_degree: float):
        """The same position angle gives the same covariance in any unit that expresses it."""
        reference = Beam(major=3 * u.arcsec, minor=1.5 * u.arcsec, pa=30 * u.deg)
        beam = Beam(major=3 * u.arcsec, minor=1.5 * u.arcsec, pa=30 * per_degree * unit)

        assert beam.pa.unit == unit
        assert np.allclose(
            beam_covariance_en(beam), beam_covariance_en(reference), rtol=1e-12, atol=0.0
        )

    @pytest.mark.parametrize(
        ("unit", "per_arcsec"),
        [
            (u.arcsec, 1.0),
            (u.arcmin, 1 / 60),
            (u.deg, 1 / 3600),
            (u.mas, 1000.0),
            (u.rad, np.pi / 648000),
        ],
        ids=str,
    )
    def test_beam_covariance_axes_in_any_angular_unit(self, unit, per_arcsec: float):
        """The major and minor axes can be in any angular unit."""
        reference = Beam(major=3 * u.arcsec, minor=1.5 * u.arcsec, pa=30 * u.deg)
        beam = Beam(major=3 * per_arcsec * unit, minor=1.5 * per_arcsec * unit, pa=30 * u.deg)

        assert beam.major.unit == unit
        assert np.allclose(
            beam_covariance_en(beam), beam_covariance_en(reference), rtol=1e-12, atol=0.0
        )

    @pytest.mark.parametrize("unit", [u.rad, u.arcmin, u.cycle], ids=str)
    def test_convolution_to_a_beam_with_the_position_angle_in_another_unit(self, unit):
        """Convolving to a beam whose position angle is in another unit gives the same image."""
        cube = _create_test_cube(x_size=31, y_size=31, vel_size=1)
        degrees = Beam(major=3 * u.arcsec, minor=2 * u.arcsec, pa=30 * u.deg)
        other = Beam(major=3 * u.arcsec, minor=2 * u.arcsec, pa=(30 * u.deg).to(unit))

        assert other.pa.unit == unit
        assert np.allclose(
            do_convolution(cube[0], other),
            do_convolution(cube[0], degrees),
            rtol=0.0,
            atol=1e-12,
        )

    def test_kernel_covariance_pixels_anisotropic_rotated(self):
        """Test kernel_covariance_pixels uses the full pixel-scale matrix.

        For an anisotropic and rotated WCS, the resulting pixel-space
        covariance should not simply be the sky covariance divided by a single
        scalar pixel scale.
        """
        cube = _create_anisotropic_wcs_cube()
        target_beam = _get_common_beam(cube.beam)

        covariance = kernel_covariance_pixels(cube[0], target_beam)

        # Off-diagonal terms should be non-zero because of the rotation, and
        # the two axes should have different variances because of the
        # anisotropic pixel scale.
        assert not np.isclose(covariance[0, 1], 0.0)
        assert not np.isclose(covariance[0, 0], covariance[1, 1])
