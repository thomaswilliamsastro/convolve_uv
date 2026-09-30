import importlib
from importlib.metadata import PackageNotFoundError, version

import convolve_uv


def test_version_matches_installed_metadata():
    assert convolve_uv.__version__ == version("convolve_uv")


def test_version_falls_back_when_package_is_not_installed(monkeypatch):
    def not_installed(distribution_name):
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr("importlib.metadata.version", not_installed)
    try:
        reloaded = importlib.reload(convolve_uv)
        assert reloaded.__version__ == "dev"
    finally:
        monkeypatch.undo()
        importlib.reload(convolve_uv)
