from datetime import datetime
from typing import Literal

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.base import Constraint


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
