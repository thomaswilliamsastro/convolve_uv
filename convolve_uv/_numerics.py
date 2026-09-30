"""Numerical helpers underlying uv-plane convolution.

This module holds the pure, per-slice numerical machinery used by
:func:`convolve_uv.convolve_uv.convolve_uv`: beam/pixel covariance
computation, the analytic Fourier transfer function, FFT-based filtering,
NaN-interpolation kernel construction, and the 2D convolution routine
(:func:`do_convolution`) that ties them together for a single image slice.

High-level orchestration over ``SpectralCube``/``Projection`` objects (looping
over channels, handling ``VaryingResolutionSpectralCube``, huge-cube memory
warnings, etc.) lives in :mod:`convolve_uv.convolve_uv`.
"""

import astropy.units as u
import numpy as np
from astropy.convolution import convolve_fft, interpolate_replace_nans
from radio_beam import Beam
from radio_beam.utils import BeamError
from spectral_cube import Projection
from spectral_cube.utils import NoBeamError

FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))

# Upper bound on the number of elements in any single 2D array sized from
# ``pad_sigma`` or from the kernel width: 2**28 float64 elements is 2 GiB per
# array (e.g. a padded 16384 x 16384 image). Anything larger is rejected up
# front with a clear error, rather than attempting an allocation that can
# exhaust memory or run effectively forever.
_MAX_ARRAY_ELEMENTS = 2**28


def _validate_pad_sigma(pad_sigma: float) -> float:
    """Return ``pad_sigma`` as a finite, non-negative float.

    Raises:
        ValueError: If ``pad_sigma`` cannot be converted to a finite number
            greater than or equal to zero.
    """
    try:
        pad_sigma = float(pad_sigma)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("pad_sigma must be a finite, non-negative number") from None
    if not np.isfinite(pad_sigma) or pad_sigma < 0:
        raise ValueError("pad_sigma must be a finite, non-negative number")
    return pad_sigma


def _check_array_size(elements: float, description: str, advice: str) -> None:
    """Raise if an array of ``elements`` elements would exceed the size limit.

    ``elements`` is a float so that arbitrarily large (even overflowing) sizes
    can be compared without first allocating anything.

    Raises:
        ValueError: If ``elements`` is greater than the maximum array size.
    """
    if elements > _MAX_ARRAY_ELEMENTS:
        raise ValueError(
            f"{description} would need about {elements:.3g} elements, which exceeds "
            f"the limit of {_MAX_ARRAY_ELEMENTS} elements per array. {advice}"
        )


# Normalize covariance inputs and reject invalid matrices before linear algebra.
def _validate_covariance(
    covariance: np.ndarray,
    context: str,
) -> np.ndarray:
    """Return a validated, symmetric covariance matrix.

    The matrix must be finite, 2x2, symmetric within floating-point tolerance,
    and positive semidefinite. Minor asymmetry is removed by symmetrizing it.

    Args:
        covariance: Matrix to validate.
        context: Description used to contextualize validation errors.

    Returns:
        The validated covariance matrix as a floating-point NumPy array.

    Raises:
        ValueError: If the matrix is malformed, non-finite, asymmetric, or not
            positive semidefinite.
    """
    try:
        covariance = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{context} must be a finite 2x2 covariance matrix") from None
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{context} must be a finite 2x2 covariance matrix")

    scale = float(np.max(np.abs(covariance)))
    symmetry_tolerance = 100 * np.finfo(float).eps * scale
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError(f"{context} must be symmetric")
    covariance = (covariance + covariance.T) / 2

    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Unable to evaluate {context}") from error
    eigenvalue_tolerance = 100 * np.finfo(float).eps * float(np.max(np.abs(eigenvalues)))
    if eigenvalues.min() < -eigenvalue_tolerance:
        raise ValueError(f"{context} must be positive semidefinite")
    return covariance


def beam_covariance_en(
    beam: Beam,
) -> np.ndarray:
    """Gaussian covariance in (east, north), in square degrees.

    Args:
        beam (Beam): The beam to compute the covariance for.

    Returns:
        np.ndarray: The 2x2 covariance matrix of the beam in (east, north) coordinates.
    """
    try:
        smaj = beam.major.to_value(u.deg) * FWHM_TO_SIGMA
        smin = beam.minor.to_value(u.deg) * FWHM_TO_SIGMA
        angle = beam.pa.to_value(u.rad)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Beam covariance requires finite angular beam values") from error
    major_hat = np.array([np.sin(angle), np.cos(angle)])
    minor_hat = np.array([np.cos(angle), -np.sin(angle)])

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cov = smaj**2 * np.outer(major_hat, major_hat) + smin**2 * np.outer(minor_hat, minor_hat)

    cov = _validate_covariance(cov, "Beam covariance")
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
    try:
        jacobian = np.asarray(cube_slice.wcs.celestial.pixel_scale_matrix, dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Unable to compute the celestial WCS pixel-scale matrix") from error
    if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
        raise ValueError("The celestial WCS pixel-scale matrix must contain finite 2x2 values")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            determinant_sign, _ = np.linalg.slogdet(jacobian)
            if determinant_sign == 0:
                raise ValueError("The celestial WCS has a singular pixel-scale matrix")
            sky_to_pix = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError as error:
        raise ValueError("The celestial WCS has a singular pixel-scale matrix") from error
    if not np.all(np.isfinite(sky_to_pix)):
        raise ValueError("The celestial WCS pixel-scale matrix has a non-finite inverse")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cov = sky_to_pix @ kernel_sky @ sky_to_pix.T

    cov = _validate_covariance(cov, "Pixel-space kernel covariance")
    return cov


def nan_interpolation_kernel(
    covariance_xy: np.ndarray,
    pad_sigma: float = 8.0,
) -> np.ndarray:
    """Build a normalized Gaussian kernel array for NaN interpolation.

    Unlike sampling a `radio_beam.Beam` with a single, isotropic pixel scale,
    this evaluates the Gaussian directly from the full 2x2 pixel-space
    covariance matrix (the same one used by :func:`transfer_function`), so it
    correctly captures anisotropic and/or rotated pixel grids.

    Args:
        covariance_xy (np.ndarray): The 2x2 covariance matrix of the kernel in pixel
            (x, y) coordinates.
        pad_sigma (float, optional): Kernel half-size, in units of the largest marginal
            standard deviation of the covariance matrix. Must be finite and
            non-negative; zero is allowed. Defaults to 8.0.

    Returns:
        np.ndarray: A square, odd-sized, unit-sum 2D Gaussian kernel array.

    Raises:
        ValueError: If the covariance is invalid or degenerate, or if the kernel
            would be larger than the maximum supported array size.
    """
    pad_sigma = _validate_pad_sigma(pad_sigma)
    covariance_xy = _validate_covariance(covariance_xy, "NaN interpolation covariance")
    eigenvalues = np.linalg.eigvalsh(covariance_xy)
    sigma_max = np.sqrt(max(float(eigenvalues.max()), 0.0))
    if sigma_max <= 0:
        raise ValueError(
            "The kernel covariance matrix is degenerate, so no NaN interpolation "
            "kernel is available"
        )
    if eigenvalues.min() <= 0:
        raise ValueError("The NaN interpolation covariance must be positive definite")

    with np.errstate(over="ignore"):
        half_size = max(float(np.ceil(pad_sigma * sigma_max)), 1.0)
    _check_array_size(
        (2.0 * half_size + 1.0) ** 2,
        "The NaN interpolation kernel",
        "Use nan_treatment='fill' to skip the kernel, or convolve to a target beam "
        "closer to the image resolution.",
    )
    half_size = int(half_size)
    y, x = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]
    coords = np.stack([x, y], axis=-1).astype(float)

    try:
        inv_cov = np.linalg.inv(covariance_xy)
    except np.linalg.LinAlgError as error:
        raise ValueError("The NaN interpolation covariance must be positive definite") from error
    exponent = -0.5 * np.einsum("...i,ij,...j->...", coords, inv_cov, coords)
    kernel = np.exp(exponent)
    kernel /= kernel.sum()

    return kernel


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
    covariance_xy = _validate_covariance(covariance_xy, "Fourier transfer covariance")
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
    """Perform the actual convolution.

    Args:
        image_slice (Projection | SpectralCube): Either a full SpectralCube instance,
            or a projection of a full 3D SpectralCube. If a full SpectralCube, then the
            cube should only have two dimensions
        target_beam (Beam): The desired circular beam to convolve to.
        boundary (str, optional): ``'wrap'`` applies periodic boundary conditions.
            ``'fill'`` pads the image to reduce wraparound effects and excludes the
            padded pixels from the valid convolution weights. Defaults to ``'fill'``.
        fill_value (float, optional): Value used to replace non-finite data when
            ``nan_treatment='fill'``. Defaults to 0.0.
        pad_sigma (float, optional): Number of kernel standard deviations to pad on each
            side when using ``boundary='fill'``. Ignored and not validated when
            ``boundary='wrap'``. For ``'fill'``, it must be finite and non-negative;
            zero is allowed. Defaults to 8.0. The padded image (and, separately, the
            NaN interpolation kernel) may not exceed 2**28 pixels; larger sizes raise
            a ``ValueError``, for example when convolving to a beam far wider than
            the pixel scale.
        nan_treatment (str, optional): The method used to handle NaNs in the input slice:

            * ``interpolate`` (default): ``NaN`` values are replaced with interpolated
              values using the kernel as an interpolation function. Note that
              if the kernel has a sum equal to zero, NaN interpolation is not
              possible and will raise an exception. If the input and target beams
              are identical, the data are returned unchanged because no
              interpolation kernel is available.
            * ``fill``: ``NaN`` values are replaced by ``fill_value`` prior to
              convolution.
        preserve_nan (bool, optional): After performing convolution, should pixels that
            were originally NaN again become NaN? Defaults to False.

    Returns:
        np.ndarray: The convolved image_slice
    """
    # Check beams are as we expect
    try:
        beam = image_slice.beam
    except (AttributeError, NoBeamError) as error:
        raise AttributeError("image_slice must have a valid beam") from error

    if not isinstance(target_beam, Beam):
        raise TypeError("Input beam must be a Beam object")

    if boundary not in {"fill", "wrap"}:
        raise ValueError("boundary must be 'fill' or 'wrap'")
    if boundary == "fill":
        pad_sigma = _validate_pad_sigma(pad_sigma)
    if nan_treatment not in {"interpolate", "fill"}:
        raise ValueError("nan_treatment must be 'interpolate' or 'fill'")

    # A zero-width kernel cannot interpolate missing values, so preserve them.
    if beam == target_beam:
        data = np.array(image_slice.unitless_filled_data[:], dtype=image_slice.dtype)
        nan_mask = np.isnan(data)
        if nan_treatment == "fill":
            data = np.where(np.isfinite(data), data, fill_value)
        if preserve_nan:
            data[nan_mask] = np.nan
        return np.asarray(data, dtype=image_slice.dtype)

    # Check the beams can be deconvolved
    try:
        target_beam.deconvolve(image_slice.beam)
    except BeamError as error:
        raise ValueError(
            "The target beam is smaller than the input beam, so cannot be deconvolved"
        ) from error

    # The full pixel-space covariance (including any anisotropy/rotation) is used both
    # for the Fourier transfer function below, and to build a matching NaN
    # interpolation kernel.
    covariance = kernel_covariance_pixels(image_slice, target_beam)
    data = image_slice.unitless_filled_data[:]
    if image_slice.mask is None:
        mask = np.ones(data.shape, dtype=bool)
    else:
        mask = image_slice.mask.include(data=data, wcs=image_slice.wcs)

    # Keep track of NaNs, in case we need to put them back in later
    nan_mask = np.isnan(data)

    # If we're filling NaNs, then do that here
    if nan_treatment == "fill":
        data = np.where(np.isfinite(data), data, fill_value)
    else:
        interpolation_kernel = nan_interpolation_kernel(covariance)
        data = interpolate_replace_nans(
            data,
            interpolation_kernel,
            convolve=convolve_fft,
        )

    # Keep track of where pixels are valid
    valid = mask & np.isfinite(data)

    pad_y = pad_x = 0
    if boundary == "fill":
        # Marginal standard deviations give a conservative axis-wise pad.
        with np.errstate(over="ignore"):
            pad_x = float(np.ceil(pad_sigma * np.sqrt(covariance[0, 0])))
            pad_y = float(np.ceil(pad_sigma * np.sqrt(covariance[1, 1])))
        _check_array_size(
            (data.shape[-2] + 2.0 * pad_y) * (data.shape[-1] + 2.0 * pad_x),
            "The padded image",
            "Reduce pad_sigma, use boundary='wrap', or convolve to a target beam "
            "closer to the image resolution.",
        )
        pad_x = int(pad_x)
        pad_y = int(pad_y)
        pad_width = [(0, 0)] * (data.ndim - 2) + [(pad_y, pad_y), (pad_x, pad_x)]
        data = np.pad(data, pad_width, mode="constant", constant_values=fill_value)
        valid = np.pad(valid, pad_width, mode="constant", constant_values=False)

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

    if cube_slice_conv.dtype != image_slice.dtype:
        cube_slice_conv = cube_slice_conv.astype(image_slice.dtype)

    return cube_slice_conv
