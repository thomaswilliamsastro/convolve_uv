import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube

from ..convolve_uv import convolve_uv

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
    data_dtype: np.dtype = np.float32,
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


def _create_test_varying_resolution_cube(
    x_size: int = 101,
    y_size: int = 101,
    vel_size: int = 10,
    unit: u.Unit | u.IrreducibleUnit = u.K,
    pix_scale: u.Quantity = 0.1 * u.arcsec,
):
    """Set up a basic test varying resolution cube for testing

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
    # with a random orientation
    beams = [
        Beam(
            major=(0.5 + (0.02 * i)) * u.arcsec,
            minor=(0.4 + (0.01 * i)) * u.arcsec,
            pa=np.random.randint(10, 30) * u.deg,
        )
        for i in range(vel_size)
    ]
    beams = Beams(beams=beams)
    data = [
        b.as_kernel(pixscale=pix_scale, x_size=x_size, y_size=y_size).array
        for b in beams
    ]
    data = np.array(data)
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
    """Get a common round beam from a single beam

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


class TestConvolveUV:

    @pytest.mark.xfail(raises=ValueError, reason="Invalid boundary value")
    def test_non_valid_boundary(self):
        """Test passing a non-valid boundary keyword"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        convolve_uv(
            image=cube,
            target_beam=common_beam,
            boundary="this_should_fail",
        )

    @pytest.mark.xfail(raises=ValueError, reason="Singular WCS")
    def test_singular_wcs(self):
        """Test passing a singular WCS"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)

        # Manually modify to a totally broken WCS
        cube.wcs.wcs.pc = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        common_beam = _get_common_beam(cube.beam)

        convolve_uv(
            image=cube,
            target_beam=common_beam,
        )

    @pytest.mark.xfail(raises=ValueError, reason="Invalid boundary value")
    def test_non_valid_nan_treatment(self):
        """Test passing a non-valid nan_treatment keyword"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        convolve_uv(
            image=cube,
            target_beam=common_beam,
            nan_treatment="this_should_fail",
        )

    @pytest.mark.xfail(raises=AttributeError, reason="Cube without beam passed")
    def test_non_valid_cube_beam(self):
        """Test passing a cube without a beam"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale, beam=None)
        common_beam = Beam(major=1 * u.arcsec, minor=1 * u.arcsec, pa=0 * u.deg)

        convolve_uv(
            image=cube,
            target_beam=common_beam,
        )

    @pytest.mark.xfail(raises=TypeError, reason="Input beam not Beam object")
    def test_non_valid_target_beam(self):
        """Test passing a non-valid target beam"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)

        convolve_uv(
            image=cube,
            target_beam="this_should_fail",
        )

    @pytest.mark.xfail(raises=ValueError, reason="Target beam smaller than input beam")
    def test_too_small_target_beam(self):
        """Test passing a too-small target beam"""

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        target_beam = Beam(0.1 * u.arcsec, 0.1 * u.arcsec, pa=0 * u.deg)

        convolve_uv(
            image=cube,
            target_beam=target_beam,
        )

    @pytest.mark.parametrize("boundary", BOUNDARY_KEYWORDS)
    def test_boundary_keywords(
        self,
        boundary: str,
    ):
        """Test passing boundary keywords

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel[np.newaxis, ...]
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("nan_treatment", NAN_TREATMENT_KEYWORDS)
    def test_nan_treatment_keywords(
        self,
        nan_treatment: str,
    ):
        """Test passing nan_treatment keywords

        Args:
            nan_treatment (str): The nan_treatment keyword to test.
        """

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        cube_conv = convolve_uv(
            image=cube, target_beam=common_beam, nan_treatment=nan_treatment
        )

        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[2], y_size=cube.shape[1]
        ).array

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel[np.newaxis, ...]
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("preserve_nan", [True, False])
    def test_preserve_nan(
        self,
        preserve_nan: bool,
    ):
        """Test preserve_nan True and False

        Args:
            preserve_nan (bool): Whether to preserve NaN values or not.
        """

        pix_scale = 0.1 * u.arcsec
        cube = _create_test_cube(pix_scale=pix_scale)
        common_beam = _get_common_beam(cube.beam)

        cube_conv = convolve_uv(
            image=cube, target_beam=common_beam, preserve_nan=preserve_nan
        )

        res = cube_conv.unitless_filled_data[:]

        # Compare to the analytic kernel
        analytic_kernel = common_beam.as_kernel(
            pixscale=pix_scale, x_size=cube.shape[2], y_size=cube.shape[1]
        ).array

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel[np.newaxis, ...]
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_cube(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a simple cube to a round beam

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel[np.newaxis, ...]
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_cube_jy_beam(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a simple cube in Jy/beam to a round beam

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel[np.newaxis, ...]
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_slice(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a cube slice to a round beam

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("common_beam_resolution", TEST_RESOLUTIONS)
    def test_convolve_varying_resolution_spectral_cube(
        self,
        common_beam_resolution: u.Quantity | Beam | None,
    ):
        """Test convolving a varying resolution cube to a common beam

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel
        ), "Convolved kernel does not match analytic kernel"

    def test_convolve_slice_same_beam(self):
        """Test convolving a cube slice to the same beam"""

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel
        ), "Convolved kernel does not match analytic kernel"

    @pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
    def test_convolve_slice_different_dtype(
        self,
        data_dtype: np.dtype,
    ):
        """Test convolving a cube slice with a different dtype from default

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

        assert (
            cube_conv.beam == common_beam
        ), "Convolved cube beam does not match target beam"
        assert np.allclose(
            res, analytic_kernel
        ), "Convolved kernel does not match analytic kernel"
