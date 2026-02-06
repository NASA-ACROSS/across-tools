from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any, Literal
from unittest.mock import patch

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    AltAz,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.coordinate import Coordinate
from across.tools.core.schemas.polygon import Polygon as ACROSSPolygon
from across.tools.core.schemas.tle import TLE
from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.ground_ephem import GroundEphemeris
from across.tools.ephemeris.tle_ephem import TLEEphemeris, compute_tle_ephemeris
from across.tools.footprint import Footprint
from across.tools.footprint.schemas import Pointing
from across.tools.visibility.constraints.base import ConstraintABC
from across.tools.visibility.constraints.earth_limb import EarthLimbConstraint
from across.tools.visibility.constraints.moon_angle import MoonAngleConstraint
from across.tools.visibility.constraints.pointing import PointingConstraint
from across.tools.visibility.constraints.saa import SAAPolygonConstraint
from across.tools.visibility.constraints.solar_system import SolarSystemConstraint
from across.tools.visibility.constraints.sun_angle import SunAngleConstraint


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
    """Fixture to provide a step_size for testing."""
    return 60


class DummyConstraint(ConstraintABC):
    """Dummy constraint for testing purposes."""

    short_name: str = "Dummy"
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


class TrueConstraint(DummyConstraint):
    """Constraint that always returns True (always violated)."""

    short_name: Literal["True"] = "True"
    name: Literal[ConstraintType.UNKNOWN] = ConstraintType.UNKNOWN

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Always return True (constraint always violated)."""
        return np.ones(len(time), dtype=bool)


class FalseConstraint(DummyConstraint):
    """Constraint that always returns False (never violated)."""

    short_name: Literal["False"] = "False"
    name: Literal[ConstraintType.UNKNOWN] = ConstraintType.UNKNOWN

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Always return False (constraint never violated)."""
        return np.zeros(len(time), dtype=bool)


class MockEphemeris(Ephemeris):
    """Mock class for testing the Ephemeris class."""

    earth_location = EarthLocation.from_geodetic(-118.2 * u.deg, 34.2 * u.deg, 100 * u.m)

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
def true_constraint() -> TrueConstraint:
    """Fixture for a TrueConstraint instance that always returns True.

    Returns:
        TrueConstraint instance
    """
    return TrueConstraint()


@pytest.fixture
def false_constraint() -> FalseConstraint:
    """Fixture for a FalseConstraint instance that always returns False.

    Returns:
        FalseConstraint instance
    """
    return FalseConstraint()


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
def test_tle_ephemeris_no_compute(
    ephemeris_begin: datetime, ephemeris_end: datetime, ephemeris_step_size: int, test_tle: TLE
) -> Ephemeris:
    """Fixture for a basic TLE Ephemeris instance."""
    return TLEEphemeris(begin=ephemeris_begin, end=ephemeris_end, step_size=ephemeris_step_size, tle=test_tle)


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


@pytest.fixture
def ground_ephemeris(ephemeris_begin: Time, ephemeris_end: Time, ephemeris_step_size: int) -> GroundEphemeris:
    """Fixture for a GroundEphemeris object with prepared data."""
    latitude = Latitude(34.2 * u.deg)
    longitude = Longitude(-118.2 * u.deg)
    height = 100 * u.m
    ephemeris = GroundEphemeris(
        ephemeris_begin, ephemeris_end, ephemeris_step_size, latitude, longitude, height
    )
    ephemeris.compute()
    return ephemeris


@pytest.fixture
def saa_poly() -> Polygon:
    """Fixture for a basic SAA polygon. This polygon is based on the Swift one."""
    return Polygon(
        [
            (39.0, -30.0),
            (36.0, -26.0),
            (28.0, -21.0),
            (6.0, -12.0),
            (-5.0, -6.0),
            (-21.0, 2.0),
            (-30.0, 3.0),
            (-45.0, 2.0),
            (-60.0, -2.0),
            (-75.0, -7.0),
            (-83.0, -10.0),
            (-87.0, -16.0),
            (-86.0, -23.0),
            (-83.0, -30.0),
        ]
    )


@pytest.fixture
def saa_polygon_constraint(saa_poly: Polygon) -> SAAPolygonConstraint:
    """Fixture for a basic SAAPolygonConstraint instance."""
    return SAAPolygonConstraint(
        polygon=saa_poly,
    )


@pytest.fixture
def az_zero_alt_forty_five_sky_coord(ground_ephemeris: Ephemeris, ephemeris_begin: datetime) -> SkyCoord:
    """Fixture for a sky coordinate at 0 deg altitude and 45 deg azimuth."""
    return SkyCoord(
        AltAz(
            alt=45 * u.deg, az=50 * u.deg, location=ground_ephemeris.earth_location, obstime=ephemeris_begin
        )
    )


@pytest.fixture
def az_eight_alt_five_sky_coord(ground_ephemeris: Ephemeris, ephemeris_begin: datetime) -> SkyCoord:
    """Fixture for a sky coordinate at 8 deg altitude and 5 deg azimuth."""
    return SkyCoord(
        AltAz(alt=8 * u.deg, az=5 * u.deg, location=ground_ephemeris.earth_location, obstime=ephemeris_begin)
    )


class MockEphemerisWithSun(Ephemeris):
    """Mock ephemeris class for solar system magnitude testing."""

    def __init__(self, sun: SkyCoord) -> None:
        """Initialize MockEphemerisWithSun.

        Args:
            sun: SkyCoord object representing sun position
        """
        self.sun = sun

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        pass


@pytest.fixture
def mock_ephemeris_with_sun(sun_coord: SkyCoord) -> MockEphemerisWithSun:
    """Fixture for a mock ephemeris with sun coordinates.

    Args:
        sun_coord: SkyCoord fixture for sun position

    Returns:
        MockEphemerisWithSun instance
    """
    sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
    return MockEphemerisWithSun(sun_array)


@pytest.fixture
def sun_coord() -> SkyCoord:
    """Create a SkyCoord instance for sun position testing."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)


@pytest.fixture
def body_coord_1au() -> SkyCoord:
    """Create a SkyCoord instance for body at 1 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)


@pytest.fixture
def body_coord_1_5au() -> SkyCoord:
    """Create a SkyCoord instance for body at 1.5 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1.5 * u.AU)


@pytest.fixture
def body_coord_5au() -> SkyCoord:
    """Create a SkyCoord instance for body at 5 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=5 * u.AU)


@pytest.fixture
def body_coord_9au() -> SkyCoord:
    """Create a SkyCoord instance for body at 9 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=9 * u.AU)


@pytest.fixture
def body_coord_19au() -> SkyCoord:
    """Create a SkyCoord instance for body at 19 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=19 * u.AU)


@pytest.fixture
def body_coord_30au() -> SkyCoord:
    """Create a SkyCoord instance for body at 30 AU."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=30 * u.AU)


@pytest.fixture
def mauna_kea_latitude() -> Latitude:
    """Mauna Kea Observatory latitude."""
    return Latitude(19.8207 * u.deg)


@pytest.fixture
def mauna_kea_longitude() -> Longitude:
    """Mauna Kea Observatory longitude."""
    return Longitude(-155.4681 * u.deg)


@pytest.fixture
def mauna_kea_height() -> u.Quantity:
    """Mauna Kea Observatory height."""
    return 4205 * u.m


@pytest.fixture
def mauna_kea_ephemeris(
    mauna_kea_latitude: Latitude,
    mauna_kea_longitude: Longitude,
    mauna_kea_height: u.Quantity,
) -> GroundEphemeris:
    """Ground ephemeris for Mauna Kea Observatory over 24 hours."""
    start_time = Time("2024-06-15T00:00:00")  # Summer solstice for longer days
    end_time = Time("2024-06-16T00:00:00")
    step_size = 300  # 5 minutes

    ephem = GroundEphemeris(
        begin=start_time.datetime,
        end=end_time.datetime,
        step_size=step_size,
        latitude=mauna_kea_latitude,
        longitude=mauna_kea_longitude,
        height=mauna_kea_height,
    )
    ephem.compute()
    return ephem


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
    from datetime import datetime, timedelta

    begin = datetime(2025, 2, 12, 0, 0, 0)
    times = [begin + timedelta(minutes=i * 5) for i in range(5)]
    return Time(times, scale="utc")


@pytest.fixture
def dummy_coord() -> SkyCoord:
    """Create a dummy SkyCoord instance for testing."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg)


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
    """Fixture for a SolarSystemConstraint with min_separation=10.0."""
    return SolarSystemConstraint(min_separation=10.0)


@pytest.fixture
def solar_system_constraint_custom() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with custom min_separation and bodies."""
    return SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])


@pytest.fixture
def solar_system_constraint_small_separation() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with small min_separation."""
    return SolarSystemConstraint(min_separation=1.0)


@pytest.fixture
def solar_system_constraint_large_separation() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with large min_separation."""
    return SolarSystemConstraint(min_separation=100.0)


@pytest.fixture
def solar_system_constraint_single_body() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with single body."""
    return SolarSystemConstraint(bodies=["mars"], min_separation=10.0)


@pytest.fixture
def solar_system_constraint_multiple_bodies() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with multiple bodies."""
    return SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)


@pytest.fixture
def solar_system_constraint_empty_bodies() -> SolarSystemConstraint:
    """Fixture for a SolarSystemConstraint with empty bodies list."""
    return SolarSystemConstraint(bodies=[])


@pytest.fixture
def slice_index() -> slice:
    """Fixture for the common slice(0, 1) used in magnitude tests."""
    return slice(0, 1)


@pytest.fixture
def mock_get_bright_stars(
    mock_bright_stars: list[tuple[SkyCoord, float]],
) -> Generator[list[tuple[SkyCoord, float]], None, None]:
    """Fixture that patches get_bright_stars to prevent internet access."""
    with patch("across.tools.visibility.constraints.bright_star.get_bright_stars") as mock:
        mock.return_value = mock_bright_stars
        yield mock


@pytest.fixture
def mock_pointing(ephemeris_begin: datetime) -> Pointing:
    """Fixture to instantiate a mock Pointing for testing."""
    return Pointing(
        footprint=Footprint(
            detectors=[
                ACROSSPolygon(
                    coordinates=[
                        Coordinate(ra=-20.0, dec=-20.0),
                        Coordinate(ra=-20.0, dec=0.0),
                        Coordinate(ra=0.0, dec=0.0),
                        Coordinate(ra=0.0, dec=-20.0),
                        Coordinate(ra=-20.0, dec=-20.0),
                    ]
                )
            ]
        ),
        start_time=ephemeris_begin,
        end_time=ephemeris_begin + timedelta(hours=1),
    )


@pytest.fixture
def pointing_constraint(mock_pointing: Pointing) -> PointingConstraint:
    """Fixture to provide an instance of EarthLimbConstraint for testing."""
    return PointingConstraint(pointings=[mock_pointing])


@pytest.fixture
def origin_sky_coord() -> SkyCoord:
    """Create a basic SkyCoord instance at the origin."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
