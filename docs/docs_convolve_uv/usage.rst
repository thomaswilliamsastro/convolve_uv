#####
Usage
#####

``convolve-uv`` relies on ``spectral-cube`` and ``radio-beam``. To convolve a cube to a target resolution, you can
do the following:

.. code-block:: python

    from convolve_uv import convolve_uv
    from spectral_cube import SpectralCube
    from radio_beam import Beam

    # Load the cube
    cube = SpectralCube.read('my_cube.fits')

    # Define the target beam
    target_beam = Beam(major=10*u.arcsec, minor=10*u.arcsec, pa=0*u.deg)

    # Convolve the cube to the target beam
    convolved_cube = convolve_uv(cube, target_beam)

    # Save the convolved cube
    convolved_cube.write('my_convolved_cube.fits')

.. HINT::

    You can also convolve to a non-round beam.

.. code-block:: python

    target_beam = Beam(major=10*u.arcsec, minor=5*u.arcsec, pa=45*u.deg)
    convolved_cube = convolve_uv(cube, target_beam)
