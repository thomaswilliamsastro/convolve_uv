############
Installation
############

``convolve_uv`` is pip-installable. To get the latest version from PyPI, run:

.. code-block:: bash

    pip install convolve_uv

For the bleeding-edge version, you can install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/thomaswilliamsastro/convolve_uv.git

Dependency compatibility
========================

``convolve-uv`` supports Python 3.12 and newer. Its core runtime dependencies
have inclusive minimum versions, declared in ``pyproject.toml``:

* Astropy 7.2.0
* NumPy 2.3.5
* radio-beam 0.3.9
* spectral-cube 0.6.7

There are currently no upper bounds on these dependencies. CI tests the latest
compatible dependency versions across the supported Python versions and
operating systems. For every supported Python version, CI also tests with
direct dependencies resolved to their lowest compatible versions and
transitive dependencies resolved to their latest compatible versions, so the
declared minimums remain exercised. Upper bounds will only be added to address
a demonstrated upstream incompatibility.
