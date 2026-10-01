"""Tests of the kernel that fills NaNs, and of filling them."""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve_fft, interpolate_replace_nans
from radio_beam import Beam
from spectral_cube import SpectralCube

from .._numerics import (
    _MIN_INTERPOLATION_WEIGHT,
    _interpolate_nans,
    do_convolution,
    nan_interpolation_kernel,
)
from .helpers import (
    BOUNDARY_KEYWORDS,
    _create_test_cube,
)


def _weighted_mean_of_valid_neighbours(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Interpolate NaNs with plain loops, as a reference for the FFT-based version.

    Each NaN becomes the kernel-weighted mean of the valid pixels around it. Pixels
    outside the image are not data, so they are not counted.

    Args:
        data (np.ndarray): The 2D array, containing NaNs.
        kernel (np.ndarray): The square, odd-sized interpolation kernel.
    """
    half = kernel.shape[0] // 2
    ny, nx = data.shape
    result = data.copy()
    for y, x in zip(*np.where(np.isnan(data)), strict=True):
        weighted_sum = total_weight = 0.0
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and np.isfinite(data[yy, xx]):
                    weight = kernel[half + dy, half + dx]
                    weighted_sum += weight * data[yy, xx]
                    total_weight += weight
        result[y, x] = weighted_sum / total_weight if total_weight > 0 else np.nan
    return result


NAN_LAYOUTS = {
    "centre": [(10, 10)],
    "edge": [(0, 10)],
    "corner": [(0, 0)],
    "corner block": [(0, 0), (0, 1), (1, 0), (1, 1)],
}


def _constant_slice(nan_pixels: list[tuple[int, int]], size: int = 21):
    """Make a projection of an all-ones image with NaNs at the given pixels."""
    cube = _create_test_cube(x_size=size, y_size=size, vel_size=1)
    data = np.ones((1, size, size))
    for y, x in nan_pixels:
        data[0, y, x] = np.nan
    cube = SpectralCube(data=data, wcs=cube.wcs, beam=cube.beam, allow_huge_operations=True)
    return cube[0]


class TestNanInterpolationKernel:
    def test_nan_interpolation_kernel_matches_covariance(self):
        """Test the NaN interpolation kernel's second moments match the input covariance.

        This directly exercises the fix for using the full pixel-space beam
        covariance (rather than a single, isotropic pixel scale) to build the
        NaN interpolation kernel, for both anisotropic and rotated cases.
        """
        covariances = [
            np.diag([9.0, 1.0]),  # anisotropic, axis-aligned
            np.array([[5.0, 3.0], [3.0, 5.0]]),  # anisotropic, rotated 45 deg
        ]

        for covariance in covariances:
            kernel = nan_interpolation_kernel(covariance)

            assert kernel.shape[0] % 2 == 1
            assert kernel.shape[1] % 2 == 1
            assert np.isclose(kernel.sum(), 1.0)

            half_y, half_x = (kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2
            y, x = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]

            measured_cov = np.array(
                [
                    [np.sum(kernel * x * x), np.sum(kernel * x * y)],
                    [np.sum(kernel * x * y), np.sum(kernel * y * y)],
                ]
            )

            assert np.allclose(measured_cov, covariance, atol=5e-2)

    def test_nan_interpolation_kernel_depends_on_full_covariance(self):
        """Test the kernel differs when only off-axis covariance terms change.

        A regression test for the bug where the interpolation kernel was built
        from ``proj_plane_pixel_scales(...)[0]`` alone, so it was blind to
        anisotropy (differing y-axis scale) and rotation (off-diagonal terms).
        """

        def _measured_covariance(kernel: np.ndarray) -> np.ndarray:
            half_y, half_x = (kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2
            y, x = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
            return np.array(
                [
                    [np.sum(kernel * x * x), np.sum(kernel * x * y)],
                    [np.sum(kernel * x * y), np.sum(kernel * y * y)],
                ]
            )

        isotropic = np.diag([4.0, 4.0])
        anisotropic = np.diag([4.0, 1.0])
        rotated = np.array([[4.0, 2.0], [2.0, 4.0]])

        cov_isotropic = _measured_covariance(nan_interpolation_kernel(isotropic))
        cov_anisotropic = _measured_covariance(nan_interpolation_kernel(anisotropic))
        cov_rotated = _measured_covariance(nan_interpolation_kernel(rotated))

        # Same [0, 0] covariance element, but the kernels must differ once the
        # rest of the covariance matrix is accounted for.
        assert not np.allclose(cov_isotropic, cov_anisotropic, atol=5e-2)
        assert not np.allclose(cov_isotropic, cov_rotated, atol=5e-2)
        assert np.allclose(cov_anisotropic, anisotropic, atol=5e-2)
        assert np.allclose(cov_rotated, rotated, atol=5e-2)

    def test_nan_interpolation_kernel_degenerate_covariance_raises(self):
        """Test a degenerate (zero) covariance matrix raises a clear error."""
        with pytest.raises(ValueError, match="degenerate"):
            nan_interpolation_kernel(np.zeros((2, 2)))

    @pytest.mark.parametrize(
        ("covariance", "message"),
        [
            (np.ones((3, 3)), "finite 2x2 covariance matrix"),
            (np.array([["bad", 0.0], [0.0, 1.0]]), "finite 2x2 covariance matrix"),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite 2x2 covariance matrix"),
            (np.array([[1.0, 0.1], [0.0, 1.0]]), "must be symmetric"),
            (np.diag([1.0, -1.0]), "must be positive semidefinite"),
            (np.diag([1.0, 0.0]), "must be positive definite"),
        ],
    )
    def test_invalid_interpolation_covariance_raises(
        self,
        covariance: np.ndarray,
        message: str,
    ):
        """Invalid and singular covariance matrices raise contextual errors."""
        with pytest.raises(ValueError, match=message):
            nan_interpolation_kernel(covariance)

    def test_covariance_eigenvalue_error_is_contextual(self, monkeypatch):
        """A covariance eigensolver error is reported with matrix context."""

        def fail_eigvalsh(_covariance):
            raise np.linalg.LinAlgError("test failure")

        monkeypatch.setattr(np.linalg, "eigvalsh", fail_eigvalsh)

        with pytest.raises(ValueError, match="Unable to evaluate NaN interpolation covariance"):
            nan_interpolation_kernel(np.eye(2))

    def test_covariance_inverse_error_is_contextual(self, monkeypatch):
        """A covariance inversion error is reported as a positive-definiteness failure."""

        def fail_inverse(_covariance):
            raise np.linalg.LinAlgError("test failure")

        monkeypatch.setattr(np.linalg, "inv", fail_inverse)

        with pytest.raises(ValueError, match="must be positive definite"):
            nan_interpolation_kernel(np.eye(2))

    def test_nan_interpolation_kernel_size_limit(self):
        """The limit is enforced by the kernel builder itself."""
        with pytest.raises(ValueError, match="The NaN interpolation kernel would need about"):
            nan_interpolation_kernel(np.eye(2) * 1e8)

    def test_max_half_size_clips_the_kernel(self):
        """A clipped kernel is smaller, still has unit sum, and keeps the values at its centre."""
        covariance = np.eye(2) * 5.0**2
        full = nan_interpolation_kernel(covariance)
        clipped = nan_interpolation_kernel(covariance, max_half_size=12)

        assert full.shape == (81, 81)
        assert clipped.shape == (25, 25)
        assert np.isclose(clipped.sum(), 1.0, rtol=0.0, atol=1e-12)
        # Same shape of Gaussian, only renormalized over the smaller area
        ratio = clipped / full[28:53, 28:53]
        assert np.allclose(ratio, ratio[0, 0], rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("max_half_size", [None, 40, 1000])
    def test_max_half_size_beyond_the_kernel_changes_nothing(self, max_half_size: int | None):
        """A clip at or beyond the kernel's own half-size leaves it as it was."""
        covariance = np.eye(2) * 5.0**2

        assert np.array_equal(
            nan_interpolation_kernel(covariance, max_half_size=max_half_size),
            nan_interpolation_kernel(covariance),
        )

    def test_max_half_size_avoids_the_size_limit(self):
        """The size limit applies to the clipped kernel, so a wide kernel can be clipped to fit."""
        kernel = nan_interpolation_kernel(np.eye(2) * 1e8, max_half_size=10)

        assert kernel.shape == (21, 21)

    def test_max_half_size_zero_gives_a_single_pixel(self):
        """The smallest clip, for a one pixel image, gives the one-pixel kernel."""
        assert np.array_equal(
            nan_interpolation_kernel(np.eye(2) * 1e4, max_half_size=0), np.ones((1, 1))
        )


class TestNanInterpolationEdges:
    """NaNs at the edge of the image are interpolated from valid pixels only."""

    TARGET_BEAM = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

    @pytest.mark.parametrize(
        "covariance",
        [np.eye(2) * 3.0**2, np.array([[9.0, 4.0], [4.0, 6.0]])],
        ids=["round", "anisotropic rotated"],
    )
    def test_matches_weighted_mean_of_valid_pixels(self, covariance: np.ndarray):
        """The interpolated values equal an independent weighted mean of valid pixels."""
        rng = np.random.default_rng(1)
        data = rng.normal(size=(21, 21)) + 5.0
        nan_pixels = [(0, 0), (0, 7), (10, 10), (20, 20), (20, 3), (9, 0), (1, 1), (15, 16)]
        for pixel in nan_pixels:
            data[pixel] = np.nan

        result = _interpolate_nans(data, covariance)
        expected = _weighted_mean_of_valid_neighbours(data, nan_interpolation_kernel(covariance))

        nan_mask = np.isnan(data)
        assert np.allclose(result[nan_mask], expected[nan_mask], rtol=1e-9, atol=0.0)
        assert np.array_equal(result[~nan_mask], data[~nan_mask])

    @staticmethod
    def _interpolate_with_unclipped_kernel(data: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Interpolate NaNs with the whole kernel, as was done before it was clipped."""
        interpolated: np.ndarray = interpolate_replace_nans(
            data,
            nan_interpolation_kernel(covariance),
            convolve=convolve_fft,
            boundary="fill",
            fill_value=np.nan,
            min_wt=_MIN_INTERPOLATION_WEIGHT,
        )
        return interpolated

    @pytest.mark.parametrize("nan_fraction", [0.05, 0.5, 0.97])
    @pytest.mark.parametrize(
        "sigma_xy",
        [(0.5, 0.5), (3.0, 3.0), (6.0, 2.0), (15.0, 15.0), (40.0, 8.0)],
        ids=lambda sigma: f"sigma {sigma[0]}x{sigma[1]}",
    )
    @pytest.mark.parametrize("shape", [(9, 12), (30, 24), (64, 40)])
    def test_clipping_the_kernel_to_the_image_changes_nothing(
        self, shape: tuple[int, int], sigma_xy: tuple[float, float], nan_fraction: float
    ):
        """The kernel is clipped to the image, which gives the same result as the whole kernel.

        Most of these have a kernel (8 sigma) bigger than the image, so they are clipped,
        and the sparse ones are close to the limit below which a NaN is not interpolated.
        """
        rng = np.random.default_rng(2)
        data = rng.normal(size=shape) + 5.0
        data[rng.random(shape) < nan_fraction] = np.nan
        covariance = np.diag(np.square(sigma_xy))

        result = _interpolate_nans(data, covariance)
        expected = self._interpolate_with_unclipped_kernel(data, covariance)

        assert np.array_equal(np.isnan(result), np.isnan(expected))
        assert np.allclose(result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_clipping_by_less_than_the_image_does_change_the_result(self):
        """The test above could pass for the wrong reason, so check that it can tell."""
        rng = np.random.default_rng(3)
        data = rng.normal(size=(30, 30)) + 5.0
        data[rng.random(data.shape) < 0.3] = np.nan
        covariance = np.eye(2) * 12.0**2
        too_small = nan_interpolation_kernel(covariance, max_half_size=5)
        too_small_result = interpolate_replace_nans(
            data,
            too_small,
            convolve=convolve_fft,
            boundary="fill",
            fill_value=np.nan,
            min_wt=_MIN_INTERPOLATION_WEIGHT,
        )

        expected = self._interpolate_with_unclipped_kernel(data, covariance)

        assert not np.allclose(too_small_result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_a_beam_far_wider_than_the_image_is_interpolated(self):
        """A kernel that would be huge, but is clipped to the image, is no longer rejected."""
        data = np.full((16, 16), np.nan)
        data[0, 0] = 7.0

        result = _interpolate_nans(data, np.eye(2) * 1e4**2)

        assert np.allclose(result, 7.0, rtol=0.0, atol=1e-9)

    def test_a_one_pixel_image_is_interpolated_without_error(self):
        """The smallest image has nothing to interpolate from, so its NaN stays."""
        assert np.isnan(_interpolate_nans(np.full((1, 1), np.nan), np.eye(2) * 4.0)).all()

    @pytest.mark.parametrize("sigma", [2.0, 5.0, 10.0])
    @pytest.mark.parametrize("layout", list(NAN_LAYOUTS))
    def test_constant_image_stays_constant(self, layout: str, sigma: float):
        """Interpolating a constant image gives that constant, wherever the NaNs are."""
        data = np.ones((21, 21))
        for pixel in NAN_LAYOUTS[layout]:
            data[pixel] = np.nan

        result = _interpolate_nans(data, np.eye(2) * sigma**2)

        assert np.allclose(result, 1.0, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    @pytest.mark.parametrize("layout", list(NAN_LAYOUTS))
    def test_convolving_a_constant_image_with_nans_gives_the_constant(
        self, layout: str, boundary: str
    ):
        """The convolved result is not darkened near NaNs at the image edge."""
        image = _constant_slice(NAN_LAYOUTS[layout])

        result = do_convolution(
            image, self.TARGET_BEAM, boundary=boundary, nan_treatment="interpolate"
        )

        assert np.allclose(result, 1.0, rtol=0.0, atol=1e-6)

    def test_nan_without_valid_data_in_reach_stays_nan(self):
        """A NaN too far from any valid pixel is not filled with a value."""
        data = np.ones((61, 61))
        data[10:51, 10:51] = np.nan

        result = _interpolate_nans(data, np.eye(2) * 2.0**2)

        assert np.isnan(result[30, 30])
        assert np.allclose(result[np.isfinite(result)], 1.0, rtol=0.0, atol=1e-9)

    def test_an_all_nan_image_stays_nan(self):
        """With no valid data at all nothing can be interpolated."""
        result = _interpolate_nans(np.full((21, 21), np.nan), np.eye(2) * 2.0**2)

        assert np.all(np.isnan(result))

    def test_a_large_hole_is_not_filled_with_zeros(self):
        """A hole bigger than the kernel's reach does not turn into zeros in the result."""
        # With this beam the kernel sigma is about 3 px, so the centre of the 41 x 41
        # hole, 20 px from the nearest valid pixel, is out of the interpolation's reach.
        target_beam = Beam(major=1.0 * u.arcsec, minor=1.0 * u.arcsec, pa=0 * u.deg)
        cube = _create_test_cube(x_size=61, y_size=61, vel_size=1)
        data = np.ones((1, 61, 61))
        data[0, 10:51, 10:51] = np.nan
        cube = SpectralCube(data=data, wcs=cube.wcs, beam=cube.beam, allow_huge_operations=True)

        result = do_convolution(cube[0], target_beam, nan_treatment="interpolate")

        assert np.isfinite(result).any()
        assert np.allclose(result[np.isfinite(result)], 1.0, rtol=0.0, atol=1e-6)
