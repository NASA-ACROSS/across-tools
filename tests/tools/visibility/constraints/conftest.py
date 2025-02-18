from datetime import datetime
from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.ground_ephem import compute_ground_ephemeris
from across.tools.visibility.constraints.base import Constraint


@pytest.fixture
def sky_coord() -> SkyCoord:
    """Create a basic SkyCoord instance."""
    return SkyCoord(ra=150 * u.deg, dec=20 * u.deg)


@pytest.fixture
def ephemeris_begin() -> datetime:
    """Fixture to provide a begin datetime for testing."""
    return datetime(2025, 2, 12, 0, 0, 0)


@pytest.fixture
def ephemeris_end() -> datetime:
    """Fixture to provide an end datetime for testing."""
    return datetime(2025, 2, 12, 0, 5, 0)


@pytest.fixture
def ephemeris_step_size() -> int:
    """Fixture to provide a step_size for testing."""
    return 60


@pytest.fixture
def keck_latitude() -> u.Quantity:
    """Fixture to provide a latitude for testing."""
    return 20.879 * u.deg


@pytest.fixture
def keck_longitude() -> u.Quantity:
    """Fixture to provide a longitude for testing."""
    return 155.6655 * u.deg


@pytest.fixture
def keck_height() -> u.Quantity:
    """Fixture to provide a height for testing."""
    return 4160 * u.m


@pytest.fixture
def keck_ground_ephemeris(
    ephemeris_begin: datetime,
    ephemeris_end: datetime,
    ephemeris_step_size: int,
    keck_latitude: u.Quantity,
    keck_longitude: u.Quantity,
    keck_height: u.Quantity,
) -> Ephemeris:
    """Fixture to provide the ground ephemeris for the Keck Observatory."""
    return compute_ground_ephemeris(
        begin=ephemeris_begin,
        end=ephemeris_end,
        step_size=ephemeris_step_size,
        latitude=keck_latitude,
        longitude=keck_longitude,
        height=keck_height,
    )


class DummyConstraint(Constraint):
    """Dummy constraint for testing purposes."""

    name: Literal["Dummy Constraint"] = "Dummy Constraint"
    short_name: Literal["dummy"] = "dummy"

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Dummy implementation of the constraint.

        Args:
            time: Time array to evaluate constraint
            ephemeris: Ephemeris object containing orbital data
            skycoord: Sky coordinates to evaluate

        Returns:
            Boolean array indicating constraint satisfaction
        """
        return np.zeros(len(time), dtype=bool)


class MockEphemeris(Ephemeris):
    """Mock class for testing the Ephemeris class."""

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        pass


@pytest.fixture
def dummy_constraint_class() -> type[DummyConstraint]:
    """Fixture for a basic DummyConstraint instance.

    Returns:
        DummyConstraint instance
    """
    return DummyConstraint


@pytest.fixture
def dummy_constraint() -> DummyConstraint:
    """Fixture for a basic DummyConstraint instance.

    Returns:
        DummyConstraint instance
    """
    return DummyConstraint()


@pytest.fixture
def dummy_constraint_with_angles() -> DummyConstraint:
    """Fixture for a DummyConstraint instance with min and max angles.

    Returns:
        DummyConstraint instance with angles
    """
    return DummyConstraint(min_angle=10, max_angle=20)


@pytest.fixture
def mock_ephemeris() -> MockEphemeris:
    """Fixture for a basic MockEphemeris instance.

    Returns:
        MockEphemeris instance
    """
    return MockEphemeris(begin=Time(datetime(2023, 1, 1)), end=Time(datetime(2023, 1, 2)), step_size=60)


@pytest.fixture
def time_array() -> Time:
    """Fixture for a Time array.

    Returns:
        Time array with two timestamps
    """
    return Time([datetime(2023, 1, 1), datetime(2023, 1, 2)])


@pytest.fixture
def scalar_time() -> Time:
    """Fixture for a scalar Time instance.

    Returns:
        Single timestamp
    """
    return Time(datetime(2023, 1, 1))
