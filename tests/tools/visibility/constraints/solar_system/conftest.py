from datetime import datetime, timedelta
from typing import Any

import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.solar_system import SolarSystemConstraint


class MockEphemerisWithSun(Ephemeris):
    """Mock ephemeris class for solar system magnitude testing."""

    def __init__(self, sun: SkyCoord) -> None:
        self.sun = sun

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        pass


@pytest.fixture
def sun_coord() -> SkyCoord:
    """Create a SkyCoord instance for sun position testing."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)


@pytest.fixture
def mock_ephemeris_with_sun(sun_coord: SkyCoord) -> MockEphemerisWithSun:
    """Fixture for a mock ephemeris with sun coordinates."""
    sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
    return MockEphemerisWithSun(sun_array)


@pytest.fixture
def body_coord_1au() -> SkyCoord:
    """Fixture for a body coordinate at 1 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)


@pytest.fixture
def body_coord_1_5au() -> SkyCoord:
    """Fixture for a body coordinate at 1.5 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1.5 * u.AU)


@pytest.fixture
def body_coord_5au() -> SkyCoord:
    """Fixture for a body coordinate at 5 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=5 * u.AU)


@pytest.fixture
def body_coord_9au() -> SkyCoord:
    """Fixture for a body coordinate at 9 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=9 * u.AU)


@pytest.fixture
def body_coord_19au() -> SkyCoord:
    """Fixture for a body coordinate at 19 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=19 * u.AU)


@pytest.fixture
def body_coord_30au() -> SkyCoord:
    """Fixture for a body coordinate at 30 AU distance."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=30 * u.AU)


@pytest.fixture
def mock_get_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock get_slice to return slice(0, 5) for testing."""
    import across.tools.visibility.constraints.solar_system as ss

    monkeypatch.setattr(ss, "get_slice", lambda time, ephem: slice(0, 5))
    monkeypatch.setattr("across.tools.visibility.constraints.base.get_slice", lambda time, ephem: slice(0, 5))


@pytest.fixture
def mock_get_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock get_body to return dummy SkyCoord for testing."""

    def mock_body(body: Any, time: Time, location: Any) -> SkyCoord:
        num = len(time) if hasattr(time, "__len__") else 1
        return SkyCoord(ra=[150] * num * u.deg, dec=[20] * num * u.deg, distance=[1.5] * num * u.AU)

    monkeypatch.setattr("astropy.coordinates.get_body", mock_body)


@pytest.fixture
def multi_time_array() -> Time:
    """Fixture for a multi-step time array used in combined tests."""
    begin = datetime(2025, 2, 12, 0, 0, 0)
    times = [begin + timedelta(minutes=i * 5) for i in range(5)]
    return Time(times, scale="utc")


@pytest.fixture
def test_coord() -> SkyCoord:
    """Fixture for test coordinate used in combined tests."""
    return SkyCoord(ra=150 * u.deg, dec=20 * u.deg)


@pytest.fixture
def test_constraint() -> SolarSystemConstraint:
    """Fixture for test constraint used in combined tests."""
    return SolarSystemConstraint(bodies=["mars", "jupiter"], min_separation=10.0)


@pytest.fixture
def solar_system_constraint() -> SolarSystemConstraint:
    """Fixture for a basic SolarSystemConstraint instance."""
    return SolarSystemConstraint()


@pytest.fixture
def solar_system_constraint_with_separation() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with custom separation."""
    return SolarSystemConstraint(min_separation=10.0)


@pytest.fixture
def solar_system_constraint_custom() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with custom bodies and separation."""
    return SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])


@pytest.fixture
def solar_system_constraint_small_separation() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with small separation."""
    return SolarSystemConstraint(min_separation=1.0)


@pytest.fixture
def solar_system_constraint_large_separation() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with large separation."""
    return SolarSystemConstraint(min_separation=100.0)


@pytest.fixture
def solar_system_constraint_single_body() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with a single body."""
    return SolarSystemConstraint(bodies=["mars"], min_separation=10.0)


@pytest.fixture
def solar_system_constraint_multiple_bodies() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with multiple bodies."""
    return SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)


@pytest.fixture
def solar_system_constraint_empty_bodies() -> SolarSystemConstraint:
    """Fixture for SolarSystemConstraint with no bodies."""
    return SolarSystemConstraint(bodies=[])


@pytest.fixture
def slice_index() -> slice:
    """Fixture for a slice index used in testing."""
    return slice(0, 1)
