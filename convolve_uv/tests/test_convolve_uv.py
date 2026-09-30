import warnings

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.wcs import WCS
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube, cube_utils

from .._numerics import (
    _interpolate_nans,
    beam_covariance_en,
    do_convolution,
    kernel_covariance_pixels,
    nan_interpolation_kernel,
)
from ..convolve_uv import LargeCubeMemoryWarning, convolve_uv

TEST_RESOLUTIONS = [
    None,
    1 * u.arcsec,
    1.5 * u.arcsec,
    Beam(major=1.5 * u.arcsec, minor=1.3 * u.arcsec, pa=45 * u.deg),
]
BOUNDARY_KEYWORDS = [
    "wrap",
    "fill",
]
NAN_TREATMENT_KEYWORDS = [
    "interpolate",
    "fill",
]
DEFAULT_BEAM = Beam(major=0.85 * u.arcsec, minor=0.65 * u.arcsec, pa=45 * u.deg)


def _create_test_cube(
    x_size: int = 101,
    y_size: int = 101,
    vel_size: int = 10,
    unit: u.Unit | u.IrreducibleUnit = u.K,
    pix_scale: u.Quantity = 0.1 * u.arcsec,
    beam: Beam | None = DEFAULT_BEAM,
    data_dtype: npt.DTypeLike = np.float32,
):
    """Set up a basic test cube for testing.

    Args:
        x_size (int): Size for the cube in x-direction.
            Defaults to 101.
        y_size (int): Size for the cube in y-direction.
            Defaults to 101.
        vel_size (int): Size for the cube in velocity direction.
            Defaults to 10.
        unit (astropy.units.Unit): Unit of the cube.
            Defaults to u.K
        pix_scale (astropy.units.Quantity): Pixel scale of the cube.
            Defaults to 0.1 * u.arcsec
        beam (Beam | None): Beam for the cube. If None, will create a cube
            without a beam. Defaults to a beam with major=0.85 arcsec,
            minor=0.65 arcsec, pa=45 deg.
        data_dtype (np.dtype): Data type for the cube. Defaults to np.float32.
    """
    # Build the kernel, or fall back to a bunch of 1s
    if beam is not None:
        kernel = beam.as_kernel(pixscale=pix_scale, x_size=x_size, y_size=y_size).array
    else:
        kernel = np.ones((y_size, x_size))

    data = np.ones((vel_size, y_size, x_size)) * kernel[np.newaxis, :] * unit

    # Create a basic World Coordinate System (WCS)
    pix_scale_deg = pix_scale.to(u.deg).value

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
    wcs.wcs.crval = [0.0, 0.0, 0.0]
    wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg, 2500]
    wcs.wcs.crpix = [0.0, 0.0, 0.0]

    # Get in the dtype
    if data.dtype != data_dtype:
        data = data.astype(data_dtype)

    # Construct the SpectralCube
    cube = SpectralCube(
        data=data,
        wcs=wcs,
        beam=beam,
        allow_huge_operations=True,
    )

    return cube


def _create_anisotropic_wcs_cube(
    x_size: int = 41,
    y_size: int = 41,
    vel_size: int = 1,
    cdelt_x: u.Quantity = 0.05 * u.arcsec,
    cdelt_y: u.Quantity = 0.1 * u.arcsec,
    rotation: u.Quantity = 30 * u.deg,
    beam: Beam = DEFAULT_BEAM,
    seed: int = 0,
):
    """Set up a test cube with anisotropic and/or rotated pixels.

    Unlike ``_create_test_cube``, ``cdelt_x`` and ``cdelt_y`` can differ and the
    pixel grid can be rotated relative to the sky, so that
    ``wcs.celestial.pixel_scale_matrix`` is neither diagonal nor a multiple of
    the identity matrix.

    Args:
        x_size (int): Size for the cube in x-direction. Defaults to 41.
        y_size (int): Size for the cube in y-direction. Defaults to 41.
        vel_size (int): Size for the cube in velocity direction. Defaults to 1.
        cdelt_x (astropy.units.Quantity): Pixel scale along x. Defaults to 0.05 arcsec.
        cdelt_y (astropy.units.Quantity): Pixel scale along y. Defaults to 0.1 arcsec.
        rotation (astropy.units.Quantity): Rotation of the pixel grid relative to the
            sky. Defaults to 30 deg.
        beam (Beam): Beam for the cube. Defaults to a beam with major=0.85 arcsec,
            minor=0.65 arcsec, pa=45 deg.
        seed (int): Seed for the random data. Defaults to 0.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(vel_size, y_size, x_size)).astype(np.float32) * u.K

    cdelt_x_deg = cdelt_x.to_value(u.deg)
    cdelt_y_deg = cdelt_y.to_value(u.deg)
    theta = rotation.to_value(u.rad)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
    wcs.wcs.crval = [0.0, 0.0, 0.0]
    wcs.wcs.cdelt = [-cdelt_x_deg, cdelt_y_deg, 2500]
    wcs.wcs.crpix = [0.0, 0.0, 0.0]
    wcs.wcs.pc = [
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]

    cube = SpectralCube(
        data=data,
        wcs=wcs,
        beam=beam,
        allow_huge_operations=True,
    )

    return cube


def _create_test_varying_resolution_cube(
    x_size: int = 101,
    y_size: int = 101,
    vel_size: int = 10,
    unit: u.Unit | u.IrreducibleUnit = u.K,
    pix_scale: u.Quantity = 0.1 * u.arcsec,
):
    """Set up a basic test varying resolution cube for testing.

    Args:
        x_size (int): Size for the cube in x-direction.
            Defaults to 101.
        y_size (int): Size for the cube in y-direction.
            Defaults to 101.
        vel_size (int): Size for the cube in velocity direction.
            Defaults to 10.
        unit (astropy.units.Unit): Unit of the cube.
            Defaults to u.K.
        pix_scale (astropy.units.Quantity): Pixel scale of the cube.
            Defaults to 0.1 * u.arcsec.
    """
    # Make a bunch of beams that increase in size with velocity channel,
    # with deterministic but varied orientations
    rng = np.random.default_rng(seed=0)
    beams = [
        Beam(
            major=(0.5 + (0.02 * i)) * u.arcsec,
            minor=(0.4 + (0.01 * i)) * u.arcsec,
            pa=rng.integers(10, 30) * u.deg,
        )
        for i in range(vel_size)
    ]
    beams = Beams(beams=beams)
    kernels = [b.as_kernel(pixscale=pix_scale, x_size=x_size, y_size=y_size).array for b in beams]
    data = np.array(kernels)
    data *= unit

    # Create a basic World Coordinate System (WCS)
    pix_scale_deg = pix_scale.to(u.deg).value

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
    wcs.wcs.crval = [0.0, 0.0, 0.0]
    wcs.wcs.cdelt = [-pix_scale_deg, pix_scale_deg, 2500]
    wcs.wcs.crpix = [0.0, 0.0, 0.0]

    # Construct the SpectralCube
    cube = VaryingResolutionSpectralCube(
        data=data,
        wcs=wcs,
        beams=beams,
        allow_huge_operations=True,
    )

    return cube


def _get_common_beam(
    beam: Beam,
) -> Beam:
    """Get a common round beam from a single beam.

    Takes the BMAJ to build a round beam, potentially
    increasing the size slightly if deconvolution errors
    occur.

    Args:
        beam (Beam): a Beam object
    """
    bmaj = beam.major.to(u.arcsec)
    common_beam = Beam(major=bmaj, minor=bmaj, pa=0 * u.deg)

    # Test we can deconvolve this, else increase the size slightly
    epsilon = 5e-4
    try:
        common_beam.deconvolve(beam)
    except BeamError:
        bmaj += epsilon * u.arcsec
    common_beam = Beam(major=bmaj, minor=bmaj, pa=0 * u.deg)

    return common_beam


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

    def test_convolve_cube_without_mask(self):
        """Test convolution when the input cube has no mask."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube.mask = None
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        cube_conv = convolve_uv(image=cube, target_beam=target_beam)

        assert cube_conv.beam == target_beam
        assert np.all(np.isfinite(cube_conv.unmasked_data[:].value))

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

    def test_convolve_cube_restores_allow_huge_operations(self):
        """Test convolve_uv does not permanently mutate a cube's allow_huge_operations."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        convolve_uv(image=cube, target_beam=target_beam)

        assert cube.allow_huge_operations is False

    def test_convolve_cube_restores_allow_huge_operations_on_error(self):
        """Test convolve_uv restores allow_huge_operations even if it raises."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match="boundary must be"):
            convolve_uv(image=cube, target_beam=target_beam, boundary="invalid")

        assert cube.allow_huge_operations is False

    def test_convolve_projection_does_not_leak_allow_huge_operations(self):
        """Test convolve_uv doesn't leave a new attribute on a Projection without one."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        assert not hasattr(image_slice, "allow_huge_operations")
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        convolve_uv(image=image_slice, target_beam=target_beam)

        assert not hasattr(image_slice, "allow_huge_operations")

    def test_convolve_projection_restores_allow_huge_operations_on_error(self):
        """Test convolve_uv restores a Projection's pre-existing allow_huge_operations on error."""
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

    def test_nan_interpolation_kernel_size_limit(self):
        """The limit is enforced by the kernel builder itself."""
        with pytest.raises(ValueError, match="The NaN interpolation kernel would need about"):
            nan_interpolation_kernel(np.eye(2) * 1e8)

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
