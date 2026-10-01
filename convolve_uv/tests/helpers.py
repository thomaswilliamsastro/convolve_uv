"""Builders shared by the test modules."""

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS
from radio_beam import Beam, Beams
from radio_beam.utils import BeamError
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube

DEFAULT_BEAM = Beam(major=0.85 * u.arcsec, minor=0.65 * u.arcsec, pa=45 * u.deg)

BOUNDARY_KEYWORDS = [
    "wrap",
    "fill",
]


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
