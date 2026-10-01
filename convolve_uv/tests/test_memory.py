"""Tests of allow_huge_operations and of the warning about loading a large cube."""

import itertools
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import astropy.units as u
import numpy as np
import pytest
from radio_beam import Beam
from spectral_cube import SpectralCube, cube_utils

from .. import LargeCubeMemoryWarning, _convolve, convolve_uv
from .._numerics import (
    do_convolution,
)
from .helpers import (
    _create_test_cube,
)


class TestHugeOperations:
    """The input cube's allow_huge_operations flag and the large cube warning."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_convolve_cube_keeps_allow_huge_operations(self, flag: bool):
        """The input's allow_huge_operations is unchanged, and the result has the same one."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = flag
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        result = convolve_uv(image=cube, target_beam=target_beam)

        assert cube.allow_huge_operations is flag
        assert result.allow_huge_operations is flag

    @pytest.mark.parametrize(
        ("target_arcsec", "error"),
        [(1.5, None), (0.1, ValueError)],
        ids=["succeeds", "raises part-way through"],
    )
    def test_convolve_cube_never_writes_allow_huge_operations(
        self, target_arcsec: float, error: type[Exception] | None
    ):
        """The input is never written to, not even temporarily, whether or not the call succeeds."""
        writes = []

        # spectral-cube ships no type information, so its cube class is Any to mypy
        class RecordingCube(SpectralCube):  # type: ignore[misc]
            def __setattr__(self, name, value):
                if getattr(self, "_recording", False):
                    writes.append(name)
                super().__setattr__(name, value)

        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        cube.__class__ = RecordingCube
        cube._recording = True
        cube._recorder_check = True
        target_beam = Beam(
            major=target_arcsec * u.arcsec, minor=target_arcsec * u.arcsec, pa=0 * u.deg
        )
        expectation = (
            pytest.raises(error, match="smaller than the input beam") if error else nullcontext()
        )

        with expectation:
            convolve_uv(image=cube, target_beam=target_beam)

        assert "_recorder_check" in writes
        assert "allow_huge_operations" not in writes

    def test_concurrent_convolutions_of_one_cube_do_not_interfere(self, monkeypatch):
        """Two overlapping calls on one cube leave its allow_huge_operations alone.

        The second call starts once the first is part-way through and finishes last, the
        order in which temporarily overriding the flag used to leave it stuck on True.
        """
        original = do_convolution
        callers = itertools.count()
        lock = threading.Lock()
        first_inside = threading.Event()
        first_done = threading.Event()
        both_inside = threading.Barrier(2, timeout=30)

        def hold(*args, **kwargs):
            with lock:
                position = next(callers)
            if position == 0:
                first_inside.set()
            both_inside.wait()
            if position == 1:
                first_done.wait(30)
            return original(*args, **kwargs)

        monkeypatch.setattr(_convolve, "do_convolution", hold)
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        cube.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(convolve_uv, image=cube, target_beam=target_beam)
            first.add_done_callback(lambda _: first_done.set())
            assert first_inside.wait(30)
            second = pool.submit(convolve_uv, image=cube, target_beam=target_beam)
            first_result, second_result = first.result(60), second.result(60)

        assert cube.allow_huge_operations is False
        assert np.array_equal(
            first_result.unmasked_data[:].value, second_result.unmasked_data[:].value
        )

    def test_convolve_projection_does_not_leak_allow_huge_operations(self):
        """Test convolve_uv doesn't leave a new attribute on a Projection without one."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        assert not hasattr(image_slice, "allow_huge_operations")
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        convolve_uv(image=image_slice, target_beam=target_beam)

        assert not hasattr(image_slice, "allow_huge_operations")

    def test_convolve_projection_keeps_allow_huge_operations_on_error(self):
        """Test convolve_uv keeps a Projection's allow_huge_operations on error."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        image_slice.allow_huge_operations = False
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.raises(ValueError, match="boundary must be"):
            convolve_uv(
                image=image_slice,
                target_beam=target_beam,
                boundary="invalid",
            )

        assert image_slice.allow_huge_operations is False

    def test_convolve_cube_below_huge_threshold_is_quiet(self):
        """Test convolve_uv does not warn for a cube below the huge-operation threshold."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        assert cube.size < cube_utils.MEMORY_THRESHOLD
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("error", LargeCubeMemoryWarning)
            convolve_uv(image=cube, target_beam=target_beam)

    def test_convolve_cube_at_huge_threshold_warns(self, monkeypatch):
        """Test convolve_uv warns once a cube's size reaches the huge-operation threshold.

        The threshold itself (``spectral_cube.cube_utils.MEMORY_THRESHOLD``) is lowered so the
        test can exercise the boundary without allocating a genuinely huge cube.
        """
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", cube.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.warns(LargeCubeMemoryWarning, match=str(cube.size)):
            convolve_uv(image=cube, target_beam=target_beam)

    def test_convolve_cube_huge_warning_does_not_mutate_allow_huge_operations(self, monkeypatch):
        """Test the large-cube warning leaves the caller's allow_huge_operations untouched."""
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=2)
        cube.allow_huge_operations = False
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", cube.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with pytest.warns(LargeCubeMemoryWarning):
            convolve_uv(image=cube, target_beam=target_beam)

        assert cube.allow_huge_operations is False

    def test_convolve_projection_never_warns_about_huge_operations(self, monkeypatch):
        """Test convolve_uv never emits the large-cube warning for a bare Projection.

        Projections don't carry an ``allow_huge_operations`` attribute in spectral-cube, so
        the huge-operation check/warning should never apply to them regardless of size.
        """
        cube = _create_test_cube(x_size=21, y_size=21, vel_size=1)
        image_slice = cube[0]
        monkeypatch.setattr(cube_utils, "MEMORY_THRESHOLD", image_slice.size)
        target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("error", LargeCubeMemoryWarning)
            convolve_uv(image=image_slice, target_beam=target_beam)
