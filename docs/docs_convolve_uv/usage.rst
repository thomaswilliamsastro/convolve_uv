#####
Usage
#####

``convolve-uv`` relies on ``spectral-cube`` and ``radio-beam``. To convolve a cube to a target resolution, you can
do the following:

.. code-block:: python

    import astropy.units as u
    from convolve_uv import convolve_uv
    from radio_beam import Beam
    from spectral_cube import SpectralCube

    cube = SpectralCube.read('my_cube.fits')
    target_beam = Beam(major=10*u.arcsec, minor=10*u.arcsec, pa=0*u.deg)
    convolved_cube = convolve_uv(cube, target_beam)
    convolved_cube.write('my_convolved_cube.fits')

.. HINT::

    To convolve to a non-round beam, run this complete example:

    .. code-block:: python

        import astropy.units as u
        from convolve_uv import convolve_uv
        from radio_beam import Beam
        from spectral_cube import SpectralCube

        cube = SpectralCube.read('my_cube.fits')
        target_beam = Beam(major=10*u.arcsec, minor=5*u.arcsec, pa=45*u.deg)
        convolved_cube = convolve_uv(cube, target_beam)
        convolved_cube.write('my_convolved_cube.fits')

.. WARNING::

    When passed a full ``SpectralCube``/``VaryingResolutionSpectralCube``, ``convolve_uv``
    materializes the entire cube (and an unsliced copy of it) into memory, so memory use can
    be substantial for large cubes. If the cube is large enough that ``spectral-cube``'s own
    ``allow_huge_operations`` safeguard would normally require you to opt in, ``convolve_uv``
    emits a ``convolve_uv.LargeCubeMemoryWarning`` rather than raising an error. Your cube is
    never modified, including its ``allow_huge_operations`` attribute, and the convolved cube
    keeps the same ``allow_huge_operations`` setting as the cube you passed in.

Choosing the target beam
------------------------

The target beam can be elliptical, as in the example above, but it must be at least as large as
the input beam in every direction, so that the input beam can be deconvolved from it. If it is
not, ``convolve_uv`` raises a ``ValueError`` ("The target beam is smaller than the input beam,
so cannot be deconvolved").

For a ``VaryingResolutionSpectralCube``, whose channels have different beams, a natural choice is
the smallest beam that every channel can be convolved to, which ``radio-beam`` provides:

.. code-block:: python

    target_beam = cube.beams.common_beam()
    convolved_cube = convolve_uv(cube, target_beam)

The result is a ``SpectralCube`` with that single beam.

Edges: ``boundary`` and ``pad_sigma``
-------------------------------------

A convolution blurs data across the edge of an image, so something has to be decided about what
lies beyond it. ``boundary`` chooses:

* ``'fill'`` (the default): nothing exists beyond the edge. The image is padded with pixels that
  carry no weight, so nothing wraps around, and each output pixel is the weighted average of the
  pixels that do exist within reach of the kernel. A uniform image stays uniform right up to the
  edge. Flux is not conserved near the edge, though: the part of the kernel beyond the edge is left
  out of the average, so a source on the edge comes out brighter than the same source in the middle
  of the image.
* ``'wrap'``: the image is treated as periodic, so blur that leaves one edge comes back in at the
  opposite edge. Total flux is conserved. Use it for data that really is periodic, or to avoid the
  padding and the memory it needs.

For ``'fill'``, ``pad_sigma`` sets how far the padding extends beyond each edge, in standard
deviations of the convolution kernel. The padding is what stops the Fourier transform, which is
periodic, from wrapping blur around the image. The default of 8.0 is generous: in a test with a
point source on the edge, ``pad_sigma=2`` already left about one part in 10\ :sup:`5` of the peak
wrapped around, and ``pad_sigma=4`` less than one part in 10\ :sup:`15`. Setting it to 0 turns the
padding off, which lets the blur wrap around as it does for ``boundary='wrap'``. Larger values use
more memory and time. ``pad_sigma`` is ignored for ``boundary='wrap'``.

The padded image, and separately the kernel used to interpolate NaN values, may not exceed
2\ :sup:`28` elements. This only matters for a target beam that is very wide compared with the pixel
scale. A larger size raises a ``ValueError`` that suggests reducing ``pad_sigma``, using
``boundary='wrap'``, or convolving to a beam closer to the resolution of the image.

Missing data: ``nan_treatment``, ``fill_value`` and ``preserve_nan``
--------------------------------------------------------------------

Masked pixels, and pixels that are ``NaN``, are missing data. Masked pixels never contribute to the
pixels around them, and the mask is kept on the convolved cube. ``nan_treatment`` chooses what
happens to ``NaN`` pixels:

* ``'interpolate'`` (the default): before the convolution, each ``NaN`` is replaced by the weighted
  mean of the valid pixels around it, using the convolution kernel as the weights. Only valid pixels
  inside the image count. A ``NaN`` with no valid data within reach of the kernel (about five kernel
  widths) stays ``NaN``.
* ``'fill'``: each ``NaN`` is replaced by ``fill_value`` (0.0 by default) before the convolution, so
  it counts as real data with that value. ``fill_value`` must be a real number. ``nan`` or ``inf`` is
  allowed too, and leaves those pixels out of the convolution instead of counting them as data.
  ``fill_value`` is not used, and not checked, with ``'interpolate'``.

The difference shows in an image of ones with a single ``NaN``. Interpolating gives back 1.0 at that
pixel, whereas filling with 0.0 treats it as a real zero, which leaves a dip there and lowers its
neighbours.

``preserve_nan`` (``False`` by default, and it must be ``True`` or ``False``) puts ``NaN`` back, after the convolution, at the pixels that
were ``NaN`` in the input, so the output has the same holes as the input. Without it, the
interpolated or filled values stay in the result. A channel that already has the target beam is
treated differently, as described in :ref:`channels-at-target-beam`.

.. code-block:: python

    convolved_cube = convolve_uv(cube, target_beam, nan_treatment='fill', fill_value=0.0,
                                 preserve_nan=True)

Convolving a single image
-------------------------

``convolve_uv`` also accepts a single 2D image, such as one channel of a cube (``cube[10]``) or a
moment map (``cube.moment0()``). It returns a ``Projection`` with the target beam. All the keywords
above apply, and there is no progress bar.

.. code-block:: python

    convolved_channel = convolve_uv(cube[10], target_beam)

Progress bar
------------

When passed a full cube, ``convolve_uv`` shows a progress bar while it convolves the channels,
but only if standard output is a terminal or an IPython console. To draw nothing, pass
``show_progress=False`` (it must be ``True`` or ``False``):

.. code-block:: python

    convolved_cube = convolve_uv(cube, target_beam, show_progress=False)

Do this when calling ``convolve_uv`` from a thread other than the main one (for example from
a thread pool, or a web server): the progress bar installs a signal handler, which Python only
allows in the main thread, so on a terminal the call fails with
``ValueError: signal only works in main thread of the main interpreter``.

.. _channels-at-target-beam:

Channels that already have the target beam
------------------------------------------

An image that already has the target beam needs no convolution, so ``convolve_uv`` returns
its data unchanged. With the default ``nan_treatment='interpolate'`` this means its ``NaN``
values are left as ``NaN``, because there is no kernel to interpolate them with, whereas
``NaN`` values in every other channel are interpolated. This matters when convolving a cube
with a varying resolution to its common beam, where the channel with the widest beam already
matches the target. Use ``nan_treatment='fill'`` to treat every channel the same way, or
``preserve_nan=True`` to put every ``NaN`` back after the convolution.
