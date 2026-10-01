"""High-level uv-plane convolution orchestration over SpectralCube/Projection.

The numerical machinery (beam/pixel covariance, transfer functions, FFT
filtering, interpolation kernel construction, and 2D convolution) lives in
:mod:`convolve_uv._numerics`.
"""

import warnings
from contextlib import nullcontext

import numpy as np
from astropy.utils.console import ProgressBar
from astropy.wcs import WcsError
from radio_beam import Beam
from spectral_cube import (
    Projection,
    SpectralCube,
    VaryingResolutionSpectralCube,
    cube_utils,
)
from spectral_cube.utils import SpectralCubeWarning

from ._numerics import _validate_convolution_arguments, do_convolution


# spectral-cube ships no type information, so its warning class is Any to mypy
class LargeCubeMemoryWarning(SpectralCubeWarning):  # type: ignore[misc]
    """Warning that a cube is large enough to need ``allow_huge_operations``.

    Warned when a cube is large enough that spectral-cube's own
    ``allow_huge_operations`` safeguard would normally require the caller to
    opt in before loading the whole cube into memory.
    """


def convolve_uv(
    image: Projection | SpectralCube | VaryingResolutionSpectralCube,
    target_beam: Beam,
    boundary: str = "fill",
    fill_value: float = 0.0,
    pad_sigma: float = 8.0,
    nan_treatment: str = "interpolate",
    preserve_nan: bool = False,
    show_progress: bool = True,
) -> Projection | SpectralCube:
    """Convolve a 2D projection to a round Gaussian beam exactly in uv space.

    The spatial Gaussian transfer function is evaluated analytically on the DFT
    grid.  No image-plane convolution kernel is sampled, so sub-pixel kernels are
    handled without the discretisation problem in ``astropy.convolution``.

    Note this will also check if the slice units are Jy/beam like, and account for that here
    so no extra renormalisation is required

    Args:
        image (Projection | SpectralCube | VaryingResolutionSpectralCube): Either a full
            SpectralCube instance, or a projection of a full 3D SpectralCube.
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
              values using the kernel as an interpolation function. Only valid pixels
              inside the image are used. A ``NaN`` with no valid data within reach of
              the kernel (about five kernel widths) stays ``NaN`` and is excluded from
              the convolution. Note that if the kernel has a sum equal to zero, NaN
              interpolation is not possible and will raise an exception. An image that
              already has the target beam is returned unchanged, because there is no
              kernel to interpolate with: its ``NaN`` values stay ``NaN``. In a cube whose
              channels have different beams, this means ``NaN`` values are interpolated
              in every channel except those already at the target beam.
            * ``fill``: ``NaN`` values are replaced by ``fill_value`` prior to
              convolution.
        preserve_nan (bool, optional): After performing convolution, should pixels that
            were originally NaN again become NaN? Defaults to False.
        show_progress (bool, optional): Show a progress bar while convolving the channels
            of a cube. The bar is drawn on standard output, and only when that is a terminal
            or an IPython console. Set this to ``False`` to draw nothing, which is required
            when calling from a thread other than the main one (the bar installs a signal
            handler, which Python only allows in the main thread). Ignored for a
            ``Projection``, which has no bar. Defaults to True.

    Returns:
        Projection | SpectralCube: The convolved Projection or SpectralCube

    Note:
        For a 3D ``SpectralCube``/``VaryingResolutionSpectralCube`` input, the
        entire cube is materialized into memory to build the convolved output
        (as well as an unsliced copy of the input), so memory use may be
        substantial. If the cube is large enough that spectral-cube's own
        ``allow_huge_operations`` safeguard would normally require the caller
        to opt in (i.e. ``cube.size >= spectral_cube.cube_utils.MEMORY_THRESHOLD``),
        a :class:`LargeCubeMemoryWarning` is emitted. The input is never
        modified, including its ``allow_huge_operations`` attribute, which
        spectral-cube only uses to guard ``apply_function`` and which nothing
        here needs. The returned cube keeps the input's setting.
    """
    # Reject a bad argument before anything is allocated or warned about, instead of
    # once per channel inside do_convolution, after the output array has been created
    pad_sigma = _validate_convolution_arguments(target_beam, boundary, pad_sigma, nan_treatment)

    # Convolving a full cube requires materializing the whole cube (and a copy
    # of it) in memory. Warn using spectral-cube's own huge-operation
    # threshold/semantics (``cube_utils.is_huge``/``MEMORY_THRESHOLD``) so
    # users get the same signal they would from spectral-cube itself, without
    # us guessing at a different threshold.
    if not isinstance(image, Projection) and cube_utils.is_huge(image):
        warnings.warn(
            "convolve_uv requires loading the entire cube into memory "
            f"({image.size} pixels), which may use substantial memory. "
            "This matches the size at which spectral-cube's own "
            "`allow_huge_operations` safeguard would normally apply.",
            LargeCubeMemoryWarning,
            stacklevel=2,
        )

    # If we're a cube, then we need to loop over each plane
    if not isinstance(image, Projection):
        n_chan = image.shape[0]

        data_conv = np.zeros(image.shape, dtype=image.unmasked_data[0, 0, 0].dtype)

        # To avoid adding in unnecessary slice info to the header,
        # take a copy of the cube
        try:
            image_copy = image._new_cube_with()
        except WcsError as error:
            raise ValueError(
                "The celestial WCS is invalid or singular and cannot be prepared for convolution"
            ) from error

        with ProgressBar(n_chan) if show_progress else nullcontext() as bar:
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
                if bar is not None:
                    bar.update()

        # If we're a VaryingResolutionSpectralCube, then we need to return a
        # SpectralCube with the new beam
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
