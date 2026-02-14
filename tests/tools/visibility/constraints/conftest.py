from collections.abc import Callable
from datetime import datetime
from typing import Literal, Protocol

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.tle import TLE
from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.ground_ephem import GroundEphemeris
from across.tools.ephemeris.tle_ephem import compute_tle_ephemeris
from across.tools.visibility.constraints.base import ConstraintABC


class MockEphemeris(Ephemeris):
    """Mock class for testing the Ephemeris class."""

    earth_location = EarthLocation.from_geodetic(-118.2 * u.deg, 34.2 * u.deg, 100 * u.m)

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        pass


class DummyConstraint(ConstraintABC):
    """Dummy constraint for testing purposes."""

    short_name: str = "Dummy"
    name: Literal[ConstraintType.UNKNOWN] = ConstraintType.UNKNOWN
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """Return all-false array for test purposes."""
        return np.zeros(len(time), dtype=bool)


class _MinAngleConstraint(Protocol):
    """Protocol for constraints with a min_angle attribute."""

    min_angle: float


@pytest.fixture
def sky_coord() -> SkyCoord:
    """Create a basic SkyCoord instance."""
    return SkyCoord(ra=150 * u.deg, dec=20 * u.deg)


@pytest.fixture
def ephemeris_begin() -> datetime:
    """Fixture to provide a begin datetime for testing."""
    return datetime(2025, 2, 12, 0, 22, 0)


@pytest.fixture
def begin_time_array(ephemeris_begin: datetime) -> Time:
    """Fixture to provide a begin time array for testing."""
    return Time([ephemeris_begin], scale="utc")


@pytest.fixture
def ephemeris_end() -> datetime:
    """Fixture to provide an end datetime for testing."""
    return datetime(2025, 2, 12, 0, 27, 0)


@pytest.fixture
def ephemeris_step_size() -> int:
    """Fixture to provide a step size for testing."""
    return 60


@pytest.fixture
def mock_ephemeris() -> MockEphemeris:
    """Fixture for a basic MockEphemeris instance."""
    return MockEphemeris(begin=Time(datetime(2023, 1, 1)), end=Time(datetime(2023, 1, 2)), step_size=60)


@pytest.fixture
def dummy_constraint() -> DummyConstraint:
    """Fixture for a basic DummyConstraint instance."""
    return DummyConstraint()


@pytest.fixture
def time_array() -> Time:
    """Fixture for a Time array."""
    return Time([datetime(2023, 1, 1), datetime(2023, 1, 2)])


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
    """Fixture for a computed TLE ephemeris instance."""
    return compute_tle_ephemeris(
        begin=ephemeris_begin,
        end=ephemeris_end,
        step_size=ephemeris_step_size,
        tle=test_tle,
    )


@pytest.fixture
def ground_ephemeris(
    ephemeris_begin: datetime, ephemeris_end: datetime, ephemeris_step_size: int
) -> GroundEphemeris:
    """Fixture for a GroundEphemeris object with prepared data."""
    latitude = Latitude(34.2 * u.deg)
    longitude = Longitude(-118.2 * u.deg)
    height = 100 * u.m

    ephemeris = GroundEphemeris(
        begin=ephemeris_begin,
        end=ephemeris_end,
        step_size=ephemeris_step_size,
        latitude=latitude,
        longitude=longitude,
        height=height,
    )
    ephemeris.compute()
    return ephemeris


@pytest.fixture
def body_constraint_result_factory() -> Callable[[ConstraintABC, SkyCoord, Ephemeris], npt.NDArray[np.bool_]]:
    """Create full-timestamp constraint results for a given body-angle constraint and coordinate."""

    def _build(
        constraint: ConstraintABC, coordinate: SkyCoord, ephemeris: Ephemeris
    ) -> npt.NDArray[np.bool_]:
        return constraint(
            time=ephemeris.timestamp,
            ephemeris=ephemeris,
            coordinate=coordinate,
        )

    return _build


@pytest.fixture
def body_outside_constraint_coord_factory() -> Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord]:
    """Create a coordinate far from the selected body for outside-constraint tests."""

    def _build(ephemeris: Ephemeris, body_name: Literal["sun", "moon"]) -> SkyCoord:
        body_coord = getattr(ephemeris, body_name)[0]
        opposite_body = body_coord.directional_offset_by(0 * u.deg, 160 * u.deg)
        return SkyCoord(ra=opposite_body.ra, dec=opposite_body.dec)

    return _build


@pytest.fixture
def body_inside_constraint_coord_factory() -> Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord]:
    """Create a coordinate colocated with the selected body for inside-constraint tests."""

    def _build(ephemeris: Ephemeris, body_name: Literal["sun", "moon"]) -> SkyCoord:
        body_coord = getattr(ephemeris, body_name)[0]
        return SkyCoord(ra=body_coord.ra, dec=body_coord.dec)

    return _build


@pytest.fixture
def body_edge_constraint_coord_factory() -> (
    Callable[
        [_MinAngleConstraint, Ephemeris, Literal["sun", "moon"], int],
        SkyCoord,
    ]
):
    """Create a coordinate near the selected body-angle constraint edge."""

    def _build(
        constraint: _MinAngleConstraint,
        ephemeris: Ephemeris,
        body_name: Literal["sun", "moon"],
        body_index: int,
    ) -> SkyCoord:
        body_coord = getattr(ephemeris, body_name)[body_index]
        return SkyCoord(
            body_coord.ra,
            body_coord.dec,
            unit="deg",
            frame="icrs",
        ).directional_offset_by(180 * u.deg, constraint.min_angle * u.deg)

    return _build
