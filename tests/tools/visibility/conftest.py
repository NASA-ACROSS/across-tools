from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.visibility.base import Visibility


class MockVisibility(Visibility):
    """Test implementation of abstract Visibility class"""

    def _constraint(self, i: int) -> ConstraintType:
        return ConstraintType.UNKNOWN

    def prepare_data(self) -> None:
        """Fake data preparation"""
        assert self.timestamp is not None
        self.inconstraint = np.zeros(len(self.timestamp), dtype=bool)


@pytest.fixture
def mock_visibility_class() -> type[MockVisibility]:
    """Return the MockVisibility class for testing"""
    return MockVisibility


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
def test_step_size_int() -> int:
    """Return a step size for testing"""
    return 60


@pytest.fixture
def test_step_size_datetime_timedelta() -> timedelta:
    """Return a step size for testing"""
    return timedelta(seconds=60)


@pytest.fixture
def mock_visibility(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time], test_step_size: TimeDelta
) -> MockVisibility:
    """Return a MockVisibility object for testing"""
    ra, dec = test_coords
    begin, end = test_time_range
    return MockVisibility(
        ra=ra,
        dec=dec,
        begin=begin,
        end=end,
        step_size=test_step_size,
        observatory_id="test_observatory_id",
        observatory_name="test_observatory_name",
    )


@pytest.fixture
def mock_visibility_step_size_int(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time], test_step_size_int: int
) -> MockVisibility:
    """Return a MockVisibility object for testing"""
    ra, dec = test_coords
    begin, end = test_time_range
    return MockVisibility(
        ra=ra,
        dec=dec,
        begin=begin,
        end=end,
        step_size=test_step_size_int,
        observatory_id="test_observatory_id",
        observatory_name="test_observatory_name",
    )


@pytest.fixture
def mock_visibility_step_size_datetime_timedelta(
    test_coords: tuple[float, float],
    test_time_range: tuple[Time, Time],
    test_step_size_datetime_timedelta: timedelta,
) -> MockVisibility:
    """Return a MockVisibility object for testing"""
    ra, dec = test_coords
    begin, end = test_time_range
    return MockVisibility(
        ra=ra,
        dec=dec,
        begin=begin,
        end=end,
        step_size=test_step_size_datetime_timedelta,
        observatory_id="test_observatory_id",
        observatory_name="test_observatory_name",
    )


@pytest.fixture
def test_time_range() -> tuple[Time, Time]:
    """Return a begin and end time for testing"""
    return Time(datetime(2023, 1, 1)), Time(datetime(2023, 1, 2))


@pytest.fixture
def bad_mock_visibility(
    test_coords: tuple[float, float], test_time_range: tuple[Time, Time]
) -> MockVisibility:
    """Return a MockVisibility object with invalid step size for testing."""
    return MockVisibility(
        ra=test_coords[0],
        dec=test_coords[1],
        begin=test_time_range[0],
        end=test_time_range[1],
        step_size=TimeDelta(-60 * u.s),
        observatory_id="test_observatory_id",
        observatory_name="test_observatory_name",
    )
