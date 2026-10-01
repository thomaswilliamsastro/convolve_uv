"""Tests that the convolved image is the right one: kernels, flux, boundaries and widths."""

import itertools

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube

from .. import convolve_uv
from .._numerics import (
    FWHM_TO_SIGMA,
)
from .helpers import (
    BOUNDARY_KEYWORDS,
    TEST_RESOLUTIONS,
    _create_anisotropic_wcs_cube,
    _create_test_cube,
    _get_common_beam,
)


class TestAccuracy:
    """The result is compared with the analytic answer."""

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

    @staticmethod
    def _beam_area_in_pixels(beam: Beam, pix_scale: u.Quantity) -> float:
        """The area of a Gaussian beam in pixels, from the formula for its full widths."""
        area = np.pi * beam.major * beam.minor / (4 * np.log(2))
        return float((area / pix_scale**2).to_value(u.dimensionless_unscaled))

    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    @pytest.mark.parametrize(
        "target_beam",
        [
            Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg),
            Beam(major=1.8 * u.arcsec, minor=1.2 * u.arcsec, pa=30 * u.deg),
        ],
        ids=["round", "elliptical rotated"],
    )
    @pytest.mark.parametrize("unit", [u.Jy / u.beam, u.K], ids=["Jy/beam", "K"])
    def test_convolution_conserves_flux(self, unit: u.Unit, target_beam: Beam, boundary: str):
        """The flux of the sources is the same after convolving, whatever the beams.

        For Jy/beam the flux is the sum of the image divided by the beam area in pixels, which
        changes with the beam, so the values have to be rescaled by the ratio of the beam areas.
        For a brightness temperature the flux is proportional to the sum of the image.
        """
        pix_scale = 0.1 * u.arcsec
        data = np.zeros((2, 101, 101))
        data[0, 50, 50] = 3.0
        data[0, 40, 62] = 1.0
        data[1, 55, 45] = 2.0
        cube = _create_test_cube(pix_scale=pix_scale, data_dtype=np.float64)
        cube = SpectralCube(
            data=data * unit, wcs=cube.wcs, beam=cube.beam, allow_huge_operations=True
        )

        res = convolve_uv(cube, target_beam, boundary=boundary).unmasked_data[:].value

        if unit == u.K:
            flux_before, flux_after = data.sum(axis=(1, 2)), res.sum(axis=(1, 2))
        else:
            flux_before = data.sum(axis=(1, 2)) / self._beam_area_in_pixels(cube.beam, pix_scale)
            flux_after = res.sum(axis=(1, 2)) / self._beam_area_in_pixels(target_beam, pix_scale)
        assert np.allclose(flux_after, flux_before, rtol=1e-6, atol=0.0)
        # The sources really were blurred, so the conservation is not a matter of nothing changing
        assert np.count_nonzero(res[0] > 0.01 * res[0].max()) > 100

    @staticmethod
    def _kernel_covariance_pixels(target_beam: Beam, input_sigma_pix: float, pix: u.Quantity):
        """The covariance of the convolution kernel in (x, y) pixels, worked out by hand.

        The input beam is round. The position angle is measured from north towards east, and
        the pixel x axis points west (RA decreases with x) while y points north.
        """
        sigma_major = (target_beam.major * FWHM_TO_SIGMA / pix).to_value(u.dimensionless_unscaled)
        sigma_minor = (target_beam.minor * FWHM_TO_SIGMA / pix).to_value(u.dimensionless_unscaled)
        pa = target_beam.pa.to_value(u.rad)
        east, north = np.sin(pa), np.cos(pa)
        major_xy = np.array([-east, north])
        minor_xy = np.array([-north, -east])
        covariance = sigma_major**2 * np.outer(major_xy, major_xy)
        covariance += sigma_minor**2 * np.outer(minor_xy, minor_xy)
        return covariance - input_sigma_pix**2 * np.eye(2)

    @staticmethod
    def _gaussian(covariance_xy: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """A unit-integral 2D Gaussian with the given (x, y) covariance, at offsets in pixels."""
        inverse = np.linalg.inv(covariance_xy)
        exponent = -0.5 * (
            inverse[0, 0] * dx**2 + 2 * inverse[0, 1] * dx * dy + inverse[1, 1] * dy**2
        )
        density: np.ndarray = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance_xy)))
        return density

    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    @pytest.mark.parametrize(
        "target_beam",
        [
            Beam(major=1.8 * u.arcsec, minor=1.8 * u.arcsec, pa=0 * u.deg),
            Beam(major=2.2 * u.arcsec, minor=1.5 * u.arcsec, pa=30 * u.deg),
        ],
        ids=["round", "elliptical rotated"],
    )
    def test_boundary_with_a_source_near_the_corner(self, target_beam: Beam, boundary: str):
        """A source near the corner is blurred across the edge (wrap) or not (fill).

        With ``wrap`` the image is periodic, so the blur that crosses an edge comes back in
        at the opposite one: the result is the sum of the Gaussian over all the shifted
        copies of the image. With ``fill`` there are no copies, so only the part inside the
        image is kept and renormalized by the weight of the Gaussian that falls inside it.
        Both are worked out here directly, from the Gaussian, not from the code.
        """
        ny = nx = 61
        y0, x0 = 3, 4
        pix_scale = 0.1 * u.arcsec
        input_beam = Beam(major=1.0 * u.arcsec, minor=1.0 * u.arcsec, pa=0 * u.deg)
        data = np.zeros((1, ny, nx))
        data[0, y0, x0] = 1.0
        cube = _create_test_cube(
            x_size=nx, y_size=ny, vel_size=1, pix_scale=pix_scale, beam=input_beam
        )
        cube = SpectralCube(data=data * u.K, wcs=cube.wcs, beam=input_beam)
        covariance = self._kernel_covariance_pixels(
            target_beam, (input_beam.major * FWHM_TO_SIGMA / pix_scale).to_value(""), pix_scale
        )
        y, x = np.mgrid[0:ny, 0:nx]

        res = convolve_uv(cube, target_beam, boundary=boundary).unmasked_data[0].value

        if boundary == "wrap":
            expected = np.sum(
                [
                    self._gaussian(covariance, y - y0 + i * ny, x - x0 + j * nx)
                    for i, j in itertools.product(range(-3, 4), repeat=2)
                ],
                axis=0,
            )
        else:
            offsets_y, offsets_x = np.mgrid[-ny : ny + 1, -nx : nx + 1]
            weights = self._gaussian(covariance, offsets_y, offsets_x)
            inside = np.array(
                [
                    [
                        weights[py + ny - (ny - 1) : py + ny + 1, px + nx - (nx - 1) : px + nx + 1]
                        for px in range(nx)
                    ]
                    for py in range(ny)
                ]
            ).sum(axis=(2, 3))
            expected = self._gaussian(covariance, y - y0, x - x0) / inside
        assert np.allclose(res, expected, rtol=0.0, atol=1e-9 * expected.max())
        # The two boundaries really are different here: the corner opposite the source is a
        # few pixels from it across the edges if the image wraps, and 9 sigma away if not
        far_corner = expected[-1, -1] / expected.max()
        assert far_corner > 0.01 if boundary == "wrap" else far_corner < 1e-3

    @pytest.mark.parametrize(
        "source_covariance",
        [
            np.array([[16.0, 0.0], [0.0, 16.0]]),
            np.array([[25.0, 0.0], [0.0, 9.0]]),
            np.array([[20.0, 6.0], [6.0, 12.0]]),
        ],
        ids=["round", "elliptical", "elliptical rotated"],
    )
    @pytest.mark.parametrize(
        "target_beam",
        [
            Beam(major=1.8 * u.arcsec, minor=1.8 * u.arcsec, pa=0 * u.deg),
            Beam(major=2.2 * u.arcsec, minor=1.5 * u.arcsec, pa=30 * u.deg),
        ],
        ids=["round", "elliptical rotated"],
    )
    def test_extended_gaussian_source_widths_add_in_quadrature(
        self, target_beam: Beam, source_covariance: np.ndarray
    ):
        """A Gaussian source comes out as a Gaussian whose covariance is the sum of two.

        The convolution kernel has the target beam's covariance minus the input beam's, so
        the source's covariance, which has the input beam in it, goes up by exactly that:
        the widths add in quadrature along each axis, not in a straight line. The image is
        compared with the Gaussian worked out directly, and so are its second moments, which
        are measured from the image and do not depend on the sampling of the Gaussian.
        """
        # Large enough that the widest result, a sigma of about 10 pixels, is 8 sigma from the
        # edges, where the image would otherwise cut off the tails of the Gaussian
        ny = nx = 161
        yc = xc = 80
        pix_scale = 0.1 * u.arcsec
        input_beam = Beam(major=1.0 * u.arcsec, minor=1.0 * u.arcsec, pa=0 * u.deg)
        y, x = np.mgrid[0:ny, 0:nx]
        data = self._gaussian(source_covariance, y - yc, x - xc)[np.newaxis]
        cube = _create_test_cube(
            x_size=nx, y_size=ny, vel_size=1, pix_scale=pix_scale, beam=input_beam
        )
        cube = SpectralCube(data=data * u.K, wcs=cube.wcs, beam=input_beam)
        kernel_covariance = self._kernel_covariance_pixels(
            target_beam, (input_beam.major * FWHM_TO_SIGMA / pix_scale).to_value(""), pix_scale
        )
        expected_covariance = source_covariance + kernel_covariance

        res = convolve_uv(cube, target_beam).unmasked_data[0].value

        expected = self._gaussian(expected_covariance, y - yc, x - xc)
        assert np.allclose(res, expected, rtol=0.0, atol=1e-6 * expected.max())
        total = res.sum()
        measured = np.array(
            [
                [(res * (x - xc) ** 2).sum(), (res * (x - xc) * (y - yc)).sum()],
                [(res * (x - xc) * (y - yc)).sum(), (res * (y - yc) ** 2).sum()],
            ]
        )
        assert np.allclose(measured / total, expected_covariance, rtol=1e-4, atol=1e-4)
        # The source was not already as wide as the result, so the test can tell them apart
        assert np.trace(expected_covariance) > 1.5 * np.trace(source_covariance)

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
