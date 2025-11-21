import json
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.tle import TLE
from across.tools.core.schemas.visibility import VisibilityWindow
from across.tools.ephemeris import Ephemeris, compute_tle_ephemeris
from across.tools.visibility import (
    EphemerisVisibility,
    JointVisibility,
    compute_ephemeris_visibility,
    compute_joint_visibility,
    constraints_from_json,
)
from across.tools.visibility.base import Visibility
from across.tools.visibility.constraints import Constraint, EarthLimbConstraint


@pytest.fixture
def test_observatory_id() -> uuid.UUID:
    """Fixture for a test observatory ID"""
    return uuid.uuid4()


@pytest.fixture
def test_observatory_name() -> str:
    """Fixture for a test observatory name"""
    return "Test Observatory"


@pytest.fixture
def test_observatory_id_2() -> uuid.UUID:
    """Fixture for another test observatory ID"""
    return uuid.uuid4()


@pytest.fixture
def test_observatory_name_2() -> str:
    """Fixture for another test observatory name"""
    return "Test Observatory 2"


@pytest.fixture
def default_step_size() -> TimeDelta:
    """Fixture for a default step size"""
    return TimeDelta(60 * u.s)


class MockVisibility(Visibility):
    """Test implementation of abstract Visibility class."""

    def _constraint(self, i: int) -> ConstraintType:
        return ConstraintType.UNKNOWN

    def prepare_data(self) -> None:
        """Fake data preparation"""
        assert self.timestamp is not None
        self.inconstraint = np.array([t.datetime.hour < 1 for t in self.timestamp], dtype=bool)


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
    test_coords: tuple[float, float],
    test_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_observatory_name: str,
    test_observatory_id: uuid.UUID,
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
        observatory_id=test_observatory_id,
        observatory_name=test_observatory_name,
    )


@pytest.fixture
def mock_visibility_step_size_int(
    test_coords: tuple[float, float],
    test_time_range: tuple[Time, Time],
    test_step_size_int: int,
    test_observatory_name: str,
    test_observatory_id: uuid.UUID,
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
        observatory_id=test_observatory_id,
        observatory_name=test_observatory_name,
    )


@pytest.fixture
def mock_visibility_step_size_datetime_timedelta(
    test_coords: tuple[float, float],
    test_time_range: tuple[Time, Time],
    test_step_size_datetime_timedelta: timedelta,
    test_observatory_name: str,
    test_observatory_id: uuid.UUID,
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
        observatory_id=test_observatory_id,
        observatory_name=test_observatory_name,
    )


@pytest.fixture
def test_time_range() -> tuple[Time, Time]:
    """Return a begin and end time for testing"""
    return Time(datetime(2023, 1, 1)), Time(datetime(2023, 1, 2))


@pytest.fixture
def noon_time(test_time_range: tuple[Time, Time]) -> Time:
    """Fixture for noon time within the test time range"""
    return Time(datetime(2023, 1, 1, 12, 0, 0))


@pytest.fixture
def midnight_time(test_time_range: tuple[Time, Time]) -> Time:
    """Fixture for midnight time within the test time range"""
    return Time(datetime(2023, 1, 1, 0, 0, 0))


@pytest.fixture
def noon_time_array(test_time_range: tuple[Time, Time]) -> Time:
    """Fixture for noon time within the test time range"""
    return Time(["2023-01-01 12:00:00", "2023-01-01 12:01:00", "2023-01-01 12:02:00"])


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
def test_visibility_time_range() -> tuple[Time, Time]:
    """Fixture for a begin and end time for testing."""
    return Time(datetime(2023, 1, 1)), Time(datetime(2023, 1, 1, 0, 10, 0))


@pytest.fixture
def test_separate_visibility_time_range() -> tuple[Time, Time]:
    """
    Fixture for a begin and end time that doesn't overlap with other windows.
    Used for joint visibility testing.
    """
    return Time(datetime(2023, 1, 1, 0, 10, 0)), Time(datetime(2023, 1, 1, 0, 15, 0))


@pytest.fixture
def test_tle_ephemeris(
    test_visibility_time_range: tuple[Time, Time], test_step_size: TimeDelta, test_tle: TLE
) -> Ephemeris:
    """Fixture for a basic TLE Ephemeris instance."""
    return compute_tle_ephemeris(
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        tle=test_tle,
    )


@pytest.fixture
def skycoord_near_limb(test_tle_ephemeris: Ephemeris) -> SkyCoord:
    """Fixture for a SkyCoord near the Earth limb."""
    sky_coord = SkyCoord(
        test_tle_ephemeris.earth[5].ra, test_tle_ephemeris.earth[5].dec, unit="deg", frame="icrs"
    ).directional_offset_by(0 * u.deg, 33 * u.deg + test_tle_ephemeris.earth_radius_angle[5])
    return sky_coord


@pytest.fixture
def test_earth_limb_constraint() -> EarthLimbConstraint:
    """Fixture for an EarthLimbConstraint instance with min and max angles."""
    return EarthLimbConstraint(min_angle=33, max_angle=170)


@pytest.fixture
def test_earth_limb_constraint_2() -> EarthLimbConstraint:
    """Fixture for another EarthLimbConstraint instance with different min/max angles"""
    return EarthLimbConstraint(min_angle=30, max_angle=165)


@pytest.fixture
def test_extreme_constraint() -> EarthLimbConstraint:
    """Fixture for an extreme EarthLimbConstraint which will provide no overlapping visibility"""
    return EarthLimbConstraint(max_angle=29)


@pytest.fixture
def test_visibility(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_earth_limb_constraint: EarthLimbConstraint,
    test_observatory_name: str,
    test_observatory_id: uuid.UUID,
) -> EphemerisVisibility:
    """Fixture for an EphemerisVisibility instance with constraints."""

    visibility = EphemerisVisibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        ephemeris=test_tle_ephemeris,
        constraints=[test_earth_limb_constraint],
        observatory_name=test_observatory_name,
        observatory_id=test_observatory_id,
    )

    return visibility


@pytest.fixture
def computed_visibility(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_earth_limb_constraint: EarthLimbConstraint,
    test_observatory_id: uuid.UUID,
    test_observatory_name: str,
) -> EphemerisVisibility:
    """Fixture that returns a computed EphemerisVisibility object."""
    return compute_ephemeris_visibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        observatory_name=test_observatory_name,
        ephemeris=test_tle_ephemeris,
        constraints=[test_earth_limb_constraint],
        observatory_id=test_observatory_id,
    )


@pytest.fixture
def computed_visibility_with_overlap(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_earth_limb_constraint_2: EarthLimbConstraint,
    test_observatory_id_2: uuid.UUID,
    test_observatory_name_2: str,
) -> EphemerisVisibility:
    """
    Fixture that returns a computed EphemerisVisibility object for the second test instrument.
    Overlaps with first instrument.
    """
    return compute_ephemeris_visibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        observatory_name=test_observatory_name_2,
        ephemeris=test_tle_ephemeris,
        constraints=[test_earth_limb_constraint_2],
        observatory_id=test_observatory_id_2,
    )


@pytest.fixture
def computed_visibility_with_no_overlap(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_extreme_constraint: EarthLimbConstraint,
    test_observatory_id_2: uuid.UUID,
    test_observatory_name_2: str,
) -> EphemerisVisibility:
    """
    Fixture that returns a computed EphemerisVisibility object for the second test instrument.
    Does not overlap with first instrument.
    """
    return compute_ephemeris_visibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        observatory_name=test_observatory_name_2,
        ephemeris=test_tle_ephemeris,
        constraints=[test_extreme_constraint],
        observatory_id=test_observatory_id_2,
    )


@pytest.fixture
def computed_joint_visibility(
    computed_visibility: EphemerisVisibility,
    computed_visibility_with_overlap: EphemerisVisibility,
    test_observatory_id: uuid.UUID,
    test_observatory_id_2: uuid.UUID,
) -> JointVisibility[EphemerisVisibility]:
    """Fixture that returns computed joint visibility windows with overlap."""
    return compute_joint_visibility(
        visibilities=[
            computed_visibility,
            computed_visibility_with_overlap,
        ],
        instrument_ids=[
            test_observatory_id,
            test_observatory_id_2,
        ],
    )


@pytest.fixture
def expected_joint_visibility_windows(
    test_visibility_time_range: tuple[Time, Time],
    test_observatory_id: uuid.UUID,
    test_observatory_name: str,
) -> list[VisibilityWindow]:
    """Fixture that provides expected joint visibility windows"""
    return [
        VisibilityWindow.model_validate(
            {
                "window": {
                    "begin": {
                        "datetime": test_visibility_time_range[0],
                        "constraint": ConstraintType.WINDOW,
                        "observatory_id": test_observatory_id,
                    },
                    "end": {
                        "datetime": test_visibility_time_range[0]
                        + timedelta(minutes=4, seconds=59, microseconds=999982),
                        "constraint": ConstraintType.EARTH,
                        "observatory_id": test_observatory_id,
                    },
                },
                "max_visibility_duration": 299,
                "constraint_reason": {
                    "start_reason": f"{test_observatory_name} {ConstraintType.WINDOW.value}",
                    "end_reason": f"{test_observatory_name} {ConstraintType.EARTH.value}",
                },
            }
        )
    ]


@pytest.fixture
def constraint_json() -> Generator[str]:
    """Fixture for a JSON representation of a constraint."""
    yield json.dumps(
        [
            {"short_name": "Sun", "name": "Sun Angle", "min_angle": 45.0},
            {"short_name": "Moon", "name": "Moon Angle", "min_angle": 21.0},
            {"short_name": "Earth", "name": "Earth Limb", "min_angle": 33.0},
        ]
    )


@pytest.fixture
def constraints_from_fixture(constraint_json: str) -> list[Constraint]:
    """Fixture that provides constraints loaded from JSON."""
    return constraints_from_json(constraint_json)
