"""
Tests for SimClock — the simulated wall-clock that patches time.time().
"""

from __future__ import annotations

import time

import pytest

from src.backtest.clock import SimClock


class TestSimClockBasic:
    """Core clock functionality."""

    def test_initial_time(self):
        clock = SimClock(start_time=1_700_000_000.0)
        assert clock.now() == 1_700_000_000.0

    def test_advance(self):
        clock = SimClock(start_time=1000.0)
        clock.advance(1001.5)
        assert clock.now() == 1001.5

    def test_advance_same_time(self):
        clock = SimClock(start_time=1000.0)
        clock.advance(1000.0)  # no-op, should not raise
        assert clock.now() == 1000.0

    def test_advance_backwards_raises(self):
        clock = SimClock(start_time=1000.0)
        clock.advance(1001.0)
        with pytest.raises(ValueError, match="cannot go backwards"):
            clock.advance(999.0)

    def test_advance_monotonic_sequence(self):
        clock = SimClock(start_time=0.0)
        for ts in [1.0, 2.0, 3.0, 100.0, 100.0, 200.5]:
            clock.advance(ts)
        assert clock.now() == 200.5


class TestSimClockPatching:
    """Monkey-patching time.time()."""

    def test_install_patches_time(self):
        real = time.time
        clock = SimClock(start_time=42.0)
        clock.install()
        try:
            assert time.time() == 42.0
            clock.advance(43.0)
            assert time.time() == 43.0
        finally:
            clock.uninstall()
        # Restored
        assert time.time is real
        assert time.time() != 42.0  # back to real time

    def test_uninstall_restores_real_time(self):
        real = time.time
        clock = SimClock(start_time=0.0)
        clock.install()
        clock.uninstall()
        assert time.time is real

    def test_double_install_is_safe(self):
        clock = SimClock(start_time=1.0)
        clock.install()
        clock.install()  # second call is no-op
        try:
            assert time.time() == 1.0
        finally:
            clock.uninstall()

    def test_double_uninstall_is_safe(self):
        clock = SimClock(start_time=1.0)
        clock.install()
        clock.uninstall()
        clock.uninstall()  # no-op

    def test_context_manager(self):
        real = time.time
        with SimClock(start_time=99.0) as clock:
            assert time.time() == 99.0
            clock.advance(100.0)
            assert time.time() == 100.0
        # Restored outside context
        assert time.time is real

    def test_context_manager_restores_on_exception(self):
        real = time.time
        try:
            with SimClock(start_time=50.0):
                assert time.time() == 50.0
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        assert time.time is real

    def test_real_time_accessor(self):
        real = time.time
        clock = SimClock(start_time=0.0)
        clock.install()
        try:
            # real_time should return the original function
            assert clock.real_time is real
            real_now = clock.real_time()
            assert real_now > 1_000_000_000  # a real epoch
        finally:
            clock.uninstall()


class TestSimClockDefaults:
    """Default factory behavior with patched time.time()."""

    def test_dataclass_default_factory(self):
        """Verify that dataclass default_factory=time.time uses sim time."""
        from dataclasses import dataclass, field

        with SimClock(start_time=12345.0):

            @dataclass
            class Foo:
                created: float = field(default_factory=time.time)

            obj = Foo()
            assert obj.created == 12345.0

    def test_repr(self):
        clock = SimClock(start_time=100.0)
        r = repr(clock)
        assert "100.0" in r
        assert "installed=False" in r
