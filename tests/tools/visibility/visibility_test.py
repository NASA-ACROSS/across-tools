from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from across.tools.visibility.base import Visibility


class TestVisibility(Visibility):
    """Test implementation of abstract Visibility class"""

    def _constraint(self, i: int) -> str:
        return "test"

    def prepare_data(self) -> None:
        """Fake data preparation"""
        if self.timestamp is None:
            raise ValueError("Timestamp not set")
        self.inconstraint = np.zeros(len(self.timestamp), dtype=bool)


@pytest.fixture
def test_time_range() -> tuple[Time, Time]:
    """Return a begin and end time for testing"""
    return Time(datetime(2023, 1, 1)), Time(datetime(2023, 1, 2))


@pytest.fixture
def test_coords() -> tuple[float, float]:
    """Return RA and Dec coordinates for testing"""
    return 100.0, 45.0


@pytest.fixture
def test_skycoord() -> SkyCoord:
    """Return a SkyCoord object for testing"""
    return SkyCoord(ra=100.0 * u.deg, dec=45.0 * u.deg)


@pytest.fixture
def test_step_size() -> TimeDelta:
    """Return a step size for testing"""
    return TimeDelta(60 * u.s)


@pytest.fixture
def visibility(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time], test_step_size: TimeDelta
) -> TestVisibility:
    """Return a TestVisibility object for testing"""
    ra, dec = test_coords
    begin, end = test_time_range
    return TestVisibility(
        ra=ra,
        dec=dec,
        begin=begin,
        end=end,
        step_size=test_step_size,
    )


def test_validate_skycoord_type(test_coords: tuple[float, float], test_time_range: tuple[Time, Time]) -> None:
    """Test that SkyCoord is created from RA and Dec"""
    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(ra=ra, dec=dec, begin=begin, end=end)
    assert isinstance(vis.skycoord, SkyCoord)


def test_validate_skycoord_ra(test_coords: tuple[float, float], test_time_range: tuple[Time, Time]) -> None:
    """Test that RA is correctly set"""
    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(ra=ra, dec=dec, begin=begin, end=end)
    assert vis.ra == ra


def test_validate_skycoord_dec(test_coords: tuple[float, float], test_time_range: tuple[Time, Time]) -> None:
    """Test that Dec is correctly set"""
    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(ra=ra, dec=dec, begin=begin, end=end)
    assert vis.dec == dec

    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(ra=ra, dec=dec, begin=begin, end=end)
    assert vis.dec == dec


def test_validate_skycoord_from_skycoord_dec(
    test_skycoord: SkyCoord, test_time_range: tuple[Time, Time]
) -> None:
    """Test that Dec is correctly set from SkyCoord"""
    begin, end = test_time_range
    vis = TestVisibility(skycoord=test_skycoord, begin=begin, end=end)
    assert vis.dec == 45.0


def test_validate_step_size_type(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time]
) -> None:
    """Test that step size is correctly set to TimeDelta"""

    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(
        ra=ra,
        dec=dec,
        begin=begin,
        end=end,
        step_size=timedelta(seconds=60),
    )
    assert isinstance(vis.step_size, TimeDelta)


def test_validate_step_size_negative(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time]
) -> None:
    """Test that step size must be positive"""
    ra, dec = test_coords
    begin, end = test_time_range
    with pytest.raises(ValueError):
        TestVisibility(
            ra=ra,
            dec=dec,
            begin=begin,
            end=end,
            step_size=TimeDelta(-60 * u.s),
        )


def test_compute_timestamp_not_none(visibility: TestVisibility) -> None:
    """Test that timestamp is set after compute"""
    visibility.compute()
    assert visibility.timestamp is not None


def test_compute_entries_not_empty(visibility: TestVisibility) -> None:
    """Test that entries are not empty after compute"""
    visibility.compute()
    assert len(visibility.entries) > 0


def test_visible_at_noon(visibility: TestVisibility) -> None:
    """Test that target is visible at noon"""
    visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))  # Noon on Jan 1
    assert visibility.visible(t) is True


def test_index_type(visibility: TestVisibility) -> None:
    """Test that index returns an integer"""
    visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))
    idx = visibility.index(t)
    assert isinstance(idx, int)


def test_index_non_negative(visibility: TestVisibility) -> None:
    """Test that index is non-negative"""
    visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))
    idx = visibility.index(t)
    assert idx >= 0


def test_index_error_without_compute(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time]
) -> None:
    """Test that index raises error without compute"""
    ra, dec = test_coords
    begin, end = test_time_range
    vis = TestVisibility(ra=ra, dec=dec, begin=begin, end=end)
    with pytest.raises(ValueError):
        vis.index(Time(datetime(2023, 1, 1, 12)))
