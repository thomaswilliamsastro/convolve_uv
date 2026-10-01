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

Progress bar
------------

When passed a full cube, ``convolve_uv`` shows a progress bar while it convolves the channels,
but only if standard output is a terminal or an IPython console. To draw nothing, pass
``show_progress=False``:

.. code-block:: python

    convolved_cube = convolve_uv(cube, target_beam, show_progress=False)

Do this when calling ``convolve_uv`` from a thread other than the main one (for example from
a thread pool, or a web server): the progress bar installs a signal handler, which Python only
allows in the main thread, so on a terminal the call fails with
``ValueError: signal only works in main thread of the main interpreter``.

Channels that already have the target beam
------------------------------------------

An image that already has the target beam needs no convolution, so ``convolve_uv`` returns
its data unchanged. With the default ``nan_treatment='interpolate'`` this means its ``NaN``
values are left as ``NaN``, because there is no kernel to interpolate them with, whereas
``NaN`` values in every other channel are interpolated. This matters when convolving a cube
with a varying resolution to its common beam, where the channel with the widest beam already
matches the target. Use ``nan_treatment='fill'`` to treat every channel the same way, or
``preserve_nan=True`` to put every ``NaN`` back after the convolution.
