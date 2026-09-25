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
