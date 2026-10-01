"""Tests of the progress bar and the show_progress keyword."""

import sys
from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
from radio_beam import Beam

from .. import convolve_uv
from .helpers import _create_test_cube, _FakeTerminal


class TestShowProgress:
    # The progress bar only draws when standard output is a terminal, so these tests make
    # sys.stdout look like one, and read back what was written to it.
    target_beam = Beam(major=1.5 * u.arcsec, minor=1.5 * u.arcsec, pa=0 * u.deg)

    def test_the_bar_is_drawn_on_a_terminal_by_default(self, monkeypatch):
        """With no keyword, a cube convolved on a terminal gets a progress bar."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)

        convolve_uv(_create_test_cube(vel_size=3), self.target_beam)

        assert "100.00%" in terminal.getvalue()

    def test_the_bar_can_be_switched_off(self, monkeypatch):
        """With show_progress=False nothing is written to a terminal."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)

        convolve_uv(_create_test_cube(vel_size=3), self.target_beam, show_progress=False)

        assert terminal.getvalue() == ""

    def test_the_bar_does_not_change_the_result(self):
        """The convolved cube is the same with and without the bar."""
        cube = _create_test_cube(vel_size=3)

        with_bar = convolve_uv(cube, self.target_beam, show_progress=True)
        without_bar = convolve_uv(cube, self.target_beam, show_progress=False)

        np.testing.assert_array_equal(with_bar.unmasked_data[:], without_bar.unmasked_data[:])

    def test_a_cube_can_be_convolved_in_a_worker_thread_on_a_terminal(self, monkeypatch):
        """Without a bar there is no signal handler to install, so any thread can convolve."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)
        cube = _create_test_cube(vel_size=3)

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(convolve_uv, cube, self.target_beam, show_progress=False).result()

        assert result.shape == cube.shape
        assert terminal.getvalue() == ""

    def test_a_projection_ignores_the_keyword(self, monkeypatch):
        """A projection has no bar, so the keyword changes nothing for it."""
        terminal = _FakeTerminal()
        monkeypatch.setattr(sys, "stdout", terminal)
        projection = _create_test_cube(vel_size=2)[0]

        shown = convolve_uv(projection, self.target_beam, show_progress=True)
        hidden = convolve_uv(projection, self.target_beam, show_progress=False)

        np.testing.assert_array_equal(shown.value, hidden.value)
        assert terminal.getvalue() == ""
