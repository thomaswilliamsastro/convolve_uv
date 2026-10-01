# Security policy

## Supported versions

`convolve_uv` is a small research package with a single maintainer, and it is still in beta (0.x). Security
fixes go into the latest release only, so please check that the problem is present in the latest version
(`pip install --upgrade convolve-uv`) before you report it.

## Reporting a vulnerability

Please do not report a vulnerability in a public issue or pull request.

Report it privately through GitHub: open the repository's
[Security tab](https://github.com/thomaswilliamsastro/convolve_uv/security) and choose **Report a
vulnerability**, or go straight to the
[reporting form](https://github.com/thomaswilliamsastro/convolve_uv/security/advisories/new).

If you cannot use GitHub, email thomas.g.williams@manchester.ac.uk with "convolve_uv security" in the
subject line.

Please include the affected versions, what the problem is and what an attacker could do with it, and the
smallest example you can that reproduces it.

This project is maintained by one person alongside other work, so replies are best effort and there is no
fixed response time. I will acknowledge your report, keep you updated, and credit you in the release notes
and the advisory unless you would rather I did not.

## What is in scope

`convolve_uv` works on arrays that are already in memory. It does not read or write files, make network
requests or run external programs, and reading a FITS file is done by `spectral-cube` and `astropy`. So the
issues most relevant to this project are:

- inputs to `convolve_uv` that crash it, hang it or make it use unbounded memory, for example a way past the
  limit of 2<sup>28</sup> elements on the arrays it allocates;
- problems in this repository's build and release workflows, which publish to PyPI.

A vulnerability in a dependency such as `astropy`, `spectral-cube`, `radio-beam` or `numpy` should be
reported to that project. Please tell me too if `convolve_uv` makes it worse.
