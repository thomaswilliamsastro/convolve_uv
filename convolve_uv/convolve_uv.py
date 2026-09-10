import astropy.units as u
import numpy as np
from astropy.convolution import convolve_fft, interpolate_replace_nans
from astropy.utils.console import ProgressBar
from astropy.wcs.utils import proj_plane_pixel_scales
from radio_beam import Beam
from radio_beam.utils import BeamError
from spectral_cube import Projection, SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.utils import NoBeamError

FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))


def beam_covariance_en(
    beam: Beam,
) -> np.ndarray:
    """Gaussian covariance in (east, north), in square degrees.

    Args:
        beam (Beam): The beam to compute the covariance for.

    Returns:
        np.ndarray: The 2x2 covariance matrix of the beam in (east, north) coordinates.
    """
    smaj = beam.major.to_value(u.deg) * FWHM_TO_SIGMA
    smin = beam.minor.to_value(u.deg) * FWHM_TO_SIGMA
    angle = beam.pa.to_value(u.rad)
    major_hat = np.array([np.sin(angle), np.cos(angle)])
    minor_hat = np.array([np.cos(angle), -np.sin(angle)])

    cov = smaj**2 * np.outer(major_hat, major_hat) + smin**2 * np.outer(
        minor_hat, minor_hat
    )

    return cov


def kernel_covariance_pixels(
    cube_slice: Projection,
    target_beam: Beam,
) -> np.ndarray:
    """Return target-minus-input covariance in pixel (x, y) coordinates.

    Args:
        cube_slice (Projection): 2D projection of a full 3D SpectralCube.
        target_beam (Beam): The desired circular beam to convolve to.

    Returns:
        np.ndarray: The 2x2 covariance matrix of the kernel in pixel coordinates.
    """

    target = beam_covariance_en(target_beam)
    source = beam_covariance_en(cube_slice.beam)
    kernel_sky = target - source

    eigenvalues, eigenvectors = np.linalg.eigh(kernel_sky)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    kernel_sky = (eigenvectors * eigenvalues) @ eigenvectors.T

    # pixel_scale_matrix maps (dx, dy) pixels to local projected (east, north)
    # degrees.  This includes rotation and unequal pixel scales.
    jacobian = np.asarray(cube_slice.wcs.celestial.pixel_scale_matrix, dtype=float)
    if jacobian.shape != (2, 2) or abs(np.linalg.det(jacobian)) < 1e-20:
        raise ValueError("The celestial WCS has a singular pixel-scale matrix")
    sky_to_pix = np.linalg.inv(jacobian)

    cov = sky_to_pix @ kernel_sky @ sky_to_pix.T

    return cov


def transfer_function(
    shape_yx: tuple[int, int],
    covariance_xy: np.ndarray,
) -> np.ndarray:
    """Analytic Fourier transform of a unit-integral Gaussian.

    Args:
        shape_yx (tuple[int, int]): The shape of the 2D array in (y, x) order.
        covariance_xy (np.ndarray): The 2x2 covariance matrix of the Gaussian in pixel coordinates.

    Returns:
        np.ndarray: The transfer function in Fourier space.
    """
    ny, nx = shape_yx
    fx = np.fft.rfftfreq(nx)
    fy = np.fft.fftfreq(ny)

    # k is cycles/pixel and covariance is in pixel^2.
    exponent = (
        -2.0
        * np.pi**2
        * (
            covariance_xy[0, 0] * fx[None, :] ** 2
            + 2.0 * covariance_xy[0, 1] * fy[:, None] * fx[None, :]
            + covariance_xy[1, 1] * fy[:, None] ** 2
        )
    )
    t_func = np.exp(exponent)

    return t_func


def fft_filter(
    data: np.ndarray,
    transfer: np.ndarray,
) -> np.ndarray:
    """Simple wrapper around the FFT-based filtering of a 2D array with a transfer function.

    Args:
        data (np.ndarray): The 2D array to be filtered.
        transfer (np.ndarray): The transfer function in Fourier space.

    Returns:
        np.ndarray: The filtered 2D array.
    """

    data_fft_filtered = np.fft.irfft2(
        np.fft.rfft2(data) * transfer,
        s=data.shape,
    )

    return data_fft_filtered


def do_convolution(
    image_slice: Projection,
    target_beam: Beam,
    boundary: str = "fill",
    fill_value: float = 0.0,
    pad_sigma: float = 8.0,
    nan_treatment: str = "interpolate",
    preserve_nan: bool = False,
) -> np.ndarray:
    """Perform the actual convolution

    Args:
        image_slice (Projection | SpectralCube): Either a full SpectralCube instance,
            or a projection of a full 3D SpectralCube. If a full SpectralCube, then the cube should only
            have two dimensions
        target_beam (Beam): The desired circular beam to convolve to.
        boundary (str, optional): ``wrap`` gives the exact periodic DFT solution. ``fill`` pads by
            ``fill_value`` for ``pad_sigma`` kernel sigmas before transforming to reduce wrapping.
        fill_value (float, optional): The value to use outside the array when using ``boundary=fill`` .
            Defaults to 0.0.
        pad_sigma (float, optional): Number of kernel sigmas to pad when using ``boundary=pad``. Defaults to 8.0.
        nan_treatment (str, optional): The method used to handle NaNs in the input slice:

            * ``interpolate`` (default): ``NaN`` values are replaced with interpolated
              values using the kernel as an interpolation function. Note that
              if the kernel has a sum equal to zero, NaN interpolation is not
              possible and will raise an exception.
            * ``fill``: ``NaN`` values are replaced by ``fill_value`` prior to
              convolution.
        preserve_nan (bool, optional): After performing convolution, should pixels that were originally NaN again
            become NaN? Defaults to False.

    Returns:
        np.ndarray: The convolved image_slice
    """

    # Check beams are as we expect
    try:
        beam = image_slice.beam
    except (AttributeError, NoBeamError):
        raise AttributeError("image_slice must have a valid beam")

    if not isinstance(target_beam, Beam):
        raise TypeError("Input beam must be a Beam object")

    # If the beams are identical, we just return the data
    if beam == target_beam:
        return image_slice

    # Check the beams can be deconvolved
    try:
        kernel = target_beam.deconvolve(image_slice.beam)
    except BeamError:
        raise ValueError(
            "The target beam is smaller than the input beam, so cannot be deconvolved"
        )

    # Pull out the pixel scale, convert the kernel to an array
    pix_scale = proj_plane_pixel_scales(image_slice.wcs.celestial)[0] * u.deg
    kernel = kernel.as_kernel(pixscale=pix_scale).array

    covariance = kernel_covariance_pixels(image_slice, target_beam)
    data = image_slice.unitless_filled_data[:]

    # Keep track of NaNs, in case we need to put them back in later
    nan_mask = np.isnan(data)

    # If we're filling NaNs, then do that here
    if nan_treatment == "fill":
        data = np.where(np.isfinite(data), data, fill_value)
    elif nan_treatment == "interpolate":
        data = interpolate_replace_nans(
            data,
            kernel,
            convolve=convolve_fft,
        )
    else:
        raise ValueError("nan_treatment must be 'interpolate' or 'fill'")

    # Keep track of where pixels are valid
    valid = np.isfinite(data)

    pad_y = pad_x = 0
    if boundary == "fill":
        # Marginal standard deviations give a conservative axis-wise pad.
        pad_x = int(np.ceil(pad_sigma * np.sqrt(covariance[0, 0])))
        pad_y = int(np.ceil(pad_sigma * np.sqrt(covariance[1, 1])))
        pad_width = [(0, 0)] * (data.ndim - 2) + [(pad_y, pad_y), (pad_x, pad_x)]
        data = np.pad(data, pad_width, mode="constant", constant_values=fill_value)
        valid = np.pad(valid, pad_width, mode="constant", constant_values=False)
    elif boundary == "wrap":
        pass
    else:
        raise ValueError("boundary must be 'fill' or 'wrap'")

    transfer = transfer_function(data.shape, covariance)
    numerator = fft_filter(np.where(valid, data, 0.0), transfer)

    # Account for NaNs. This is essentially to get around numerical issues
    denominator = fft_filter(valid.astype(float), transfer)
    scale = max(float(transfer.mean()), 1e-14)
    cube_slice_conv = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-12 * scale,
    )

    # If we've padded, then back out this pad
    if boundary == "fill":
        ys = slice(pad_y, -pad_y or None)
        xs = slice(pad_x, -pad_x or None)
        cube_slice_conv = cube_slice_conv[ys, xs]

    # If we're preserving NaNs, then put them back in
    if preserve_nan:
        cube_slice_conv[nan_mask] = np.nan

    # If we're Jy/beam-like, account for that here
    if image_slice.unit.is_equivalent(u.Jy / u.beam):
        beam_ratio_factor = (target_beam.sr / image_slice.beam.sr).value
    else:
        beam_ratio_factor = 1.0
    cube_slice_conv *= beam_ratio_factor

    # If the dtype has changed, then revert here
    if cube_slice_conv.dtype != image_slice.dtype:
        cube_slice_conv = cube_slice_conv.astype(image_slice.dtype)

    return cube_slice_conv


def convolve_uv(
    image: Projection | SpectralCube | VaryingResolutionSpectralCube,
    target_beam: Beam,
    boundary: str = "fill",
    fill_value: float = 0.0,
    pad_sigma: float = 8.0,
    nan_treatment: str = "interpolate",
    preserve_nan: bool = False,
) -> Projection | SpectralCube:
    """Convolve a 2D projection to a round Gaussian beam exactly in uv space.

    The spatial Gaussian transfer function is evaluated analytically on the DFT
    grid.  No image-plane convolution kernel is sampled, so sub-pixel kernels are
    handled without the discretisation problem in ``astropy.convolution``.

    Note this will also check if the slice units are Jy/beam like, and account for that here
    so no extra renormalisation is required

    Args:
        image (Projection | SpectralCube | VaryingResolutionSpectralCube): Either a full SpectralCube instance,
            or a projection of a full 3D SpectralCube.
        target_beam (Beam): The desired circular beam to convolve to.
        boundary (str, optional): ``wrap`` gives the exact periodic DFT solution. ``fill`` pads by
            ``fill_value`` for ``pad_sigma`` kernel sigmas before transforming to reduce wrapping.
        fill_value (float, optional): The value to use outside the array when using ``boundary=fill`` .
            Defaults to 0.0.
        pad_sigma (float, optional): Number of kernel sigmas to pad when using ``boundary=pad``. Defaults to 8.0.
        nan_treatment (str, optional): The method used to handle NaNs in the input slice:

            * ``interpolate`` (default): ``NaN`` values are replaced with interpolated
              values using the kernel as an interpolation function. Note that
              if the kernel has a sum equal to zero, NaN interpolation is not
              possible and will raise an exception.
            * ``fill``: ``NaN`` values are replaced by ``fill_value`` prior to
              convolution.
        preserve_nan (bool, optional): After performing convolution, should pixels that were originally NaN again
            become NaN? Defaults to False.

    Returns:
        Projection | SpectralCube: The convolved Projection or SpectralCube
    """

    # We need to keep everything in memory
    image.allow_huge_operations = True

    # If we're a cube, then we need to loop over each plane
    if not isinstance(image, Projection):
        n_chan = image.shape[0]

        data_conv = np.zeros(image.shape, dtype=image.unmasked_data[0, 0, 0].dtype)

        # To avoid adding in unnecessary slice info to the header,
        # take a copy of the cube
        image_copy = image._new_cube_with()

        with ProgressBar(n_chan) as bar:
            for chan in range(n_chan):
                data_conv[chan] = do_convolution(
                    image_copy[chan],
                    target_beam=target_beam,
                    boundary=boundary,
                    fill_value=fill_value,
                    pad_sigma=pad_sigma,
                    nan_treatment=nan_treatment,
                    preserve_nan=preserve_nan,
                )
                bar.update()

        # If we're a VaryingResolutionSpectralCube, then we need to return a SpectralCube with the new beam
        if isinstance(image, VaryingResolutionSpectralCube):
            image_conv = SpectralCube(
                data=data_conv,
                wcs=image.wcs,
                mask=image.mask,
                meta=image.meta,
                fill_value=image.fill_value,
                header=image.header,
                beam=target_beam,
            )

        else:
            image_conv = image._new_cube_with(data=data_conv, beam=target_beam)

    else:
        slice_conv = do_convolution(
            image,
            target_beam=target_beam,
            boundary=boundary,
            fill_value=fill_value,
            pad_sigma=pad_sigma,
            nan_treatment=nan_treatment,
            preserve_nan=preserve_nan,
        )

        image_conv = image._new_projection_with(data=slice_conv, beam=target_beam)

    # Since we've convolved to a beam, if there's still references to multibeam tables,
    # remove that
    if "CASAMBM" in image_conv.header:
        del image_conv._header["CASAMBM"]

    return image_conv
