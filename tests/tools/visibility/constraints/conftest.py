from datetime import datetime
from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.tle import TLE
from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.tle_ephem import compute_tle_ephemeris
from across.tools.visibility.constraints.base import ConstraintABC
from across.tools.visibility.constraints.earth_limb import EarthLimbConstraint
from across.tools.visibility.constraints.moon_angle import MoonAngleConstraint
from across.tools.visibility.constraints.sun_angle import SunAngleConstraint


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


class DummyConstraint(ConstraintABC):
    """Dummy constraint for testing purposes."""

    short_name: Literal["Dummy"] = "Dummy"
    name: Literal[ConstraintType.UNKNOWN] = ConstraintType.UNKNOWN
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Dummy implementation of the constraint.

        Args:
            time: Time array to evaluate constraint
            ephemeris: Ephemeris object containing orbital data
            coordinate: Sky coordinates to evaluate

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
def dummy_constraint() -> DummyConstraint:
    """Fixture for a basic DummyConstraint instance.

    Returns:
        DummyConstraint instance
    """
    return DummyConstraint()


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


@pytest.fixture
def test_tle() -> TLE:
    """Fixture for a basic TLE instance."""
    tle_dict = {
        "norad_id": 28485,
        "tle1": "1 28485U 04047A   25042.85297680  .00022673  00000-0  70501-3 0  9996",
        "tle2": "2 28485  20.5544 250.6462 0005903 156.1246 203.9467 15.31282606110801",
    }
    return TLE.model_validate(tle_dict)


@pytest.fixture
def test_tle_ephemeris(
    ephemeris_begin: datetime, ephemeris_end: datetime, ephemeris_step_size: int, test_tle: TLE
) -> Ephemeris:
    """Fixture for a basic TLE Ephemeris instance."""
    return compute_tle_ephemeris(
        begin=ephemeris_begin, end=ephemeris_end, step_size=ephemeris_step_size, tle=test_tle
    )


@pytest.fixture
def moon_angle_constraint() -> MoonAngleConstraint:
    """Fixture to provide an instance of MoonAngleConstraint for testing."""
    return MoonAngleConstraint(min_angle=21.0, max_angle=170.0)


@pytest.fixture
def sun_angle_constraint() -> SunAngleConstraint:
    """Fixture to provide an instance of SunAngleConstraint for testing."""
    return SunAngleConstraint(min_angle=45.0, max_angle=170.0)


@pytest.fixture
def earth_limb_constraint() -> EarthLimbConstraint:
    """Fixture to provide an instance of EarthLimbConstraint for testing."""
    return EarthLimbConstraint(min_angle=33.0, max_angle=170.0)
