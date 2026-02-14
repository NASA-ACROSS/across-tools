import contextlib
import json
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.table import Table  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

import across.tools.visibility.catalogs as catalogs_module
from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.tle import TLE
from across.tools.core.schemas.visibility import VisibilityComputedValues, VisibilityWindow
from across.tools.ephemeris import Ephemeris, compute_tle_ephemeris
from across.tools.visibility import (
    EphemerisVisibility,
    JointVisibility,
    compute_ephemeris_visibility,
    compute_joint_visibility,
    constraints_from_json,
)
from across.tools.visibility.base import Visibility
from across.tools.visibility.catalogs import cache_clear
from across.tools.visibility.constraints import AllConstraint, EarthLimbConstraint, SunAngleConstraint
from across.tools.visibility.constraints.base import ConstraintABC


@pytest.fixture
def isolated_star_cache(tmp_path: Path) -> Generator[None, None, None]:
    """Fixture to isolate the star cache for testing."""
    cache_dir = tmp_path / "star_catalogs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=cache_dir):
        cache_clear()
        yield
        cache_clear()
        # Explicitly close any open cache connections
        if catalogs_module._cache is not None:
            with contextlib.suppress(Exception):
                catalogs_module._cache.close()
            catalogs_module._cache = None


@pytest.fixture
def test_observatory_id() -> uuid.UUID:
    """Fixture for a test observatory ID"""
    return uuid.uuid4()


@pytest.fixture
def test_observatory_name() -> str:
    """Fixture for a test observatory name"""
    return "Test Observatory"


class MockConstraint:
    """Mock constraint for testing computed values merging."""

    def __init__(self, constraint_type: ConstraintType, value_attr: str):
        """Initialize mock constraint with type and computed value attribute."""

        self.name = constraint_type
        self.computed_values = VisibilityComputedValues()

        # Set appropriate test values based on the field type
        if value_attr in ["sun_angle", "moon_angle", "earth_angle"]:
            setattr(self.computed_values, value_attr, 45.0 * u.deg)
        elif value_attr == "alt_az":
            setattr(self.computed_values, value_attr, SkyCoord(ra=0 * u.deg, dec=0 * u.deg))

    def __call__(self, *args, **kwargs):  # type: ignore
        """Mock the constraint call to return a boolean array."""
        return np.array([False] * 10, dtype=bool)


@pytest.fixture
def mock_constraint_class() -> type[MockConstraint]:
    """Return the MockConstraint class for testing"""
    return MockConstraint


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

    def _merge_computed_values(self) -> None:
        """Fake merging of computed values"""
        return

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
def computed_visibility_with_sequence_constraints(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_earth_limb_constraint: EarthLimbConstraint,
    test_observatory_id: uuid.UUID,
    test_observatory_name: str,
) -> EphemerisVisibility:
    """Fixture that returns computed visibility using Sequence constraints (tuple)."""
    return compute_ephemeris_visibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        observatory_name=test_observatory_name,
        ephemeris=test_tle_ephemeris,
        constraints=(test_earth_limb_constraint,),
        observatory_id=test_observatory_id,
    )


@pytest.fixture
def visibility_with_sequence_constraints(
    skycoord_near_limb: SkyCoord,
    test_visibility_time_range: tuple[Time, Time],
    test_step_size: TimeDelta,
    test_tle_ephemeris: Ephemeris,
    test_earth_limb_constraint: EarthLimbConstraint,
    test_observatory_id: uuid.UUID,
    test_observatory_name: str,
) -> EphemerisVisibility:
    """Fixture that returns EphemerisVisibility initialized with Sequence constraints (tuple)."""
    return EphemerisVisibility(
        coordinate=skycoord_near_limb,
        begin=test_visibility_time_range[0],
        end=test_visibility_time_range[1],
        step_size=test_step_size,
        ephemeris=test_tle_ephemeris,
        constraints=(test_earth_limb_constraint,),  # type: ignore[arg-type]
        observatory_name=test_observatory_name,
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
def boundary_visibilities(
    mock_visibility_class: type[Visibility],
    test_time_range: tuple[Time, Time],
    test_coords: tuple[float, float],
    test_step_size: TimeDelta,
    test_observatory_id: uuid.UUID,
    test_observatory_id_2: uuid.UUID,
    test_observatory_name: str,
    test_observatory_name_2: str,
) -> tuple[Visibility, Visibility]:
    """Fixture for prepared visibilities with a boundary-ending inconstraint pattern."""
    vis_1 = mock_visibility_class(
        ra=test_coords[0],
        dec=test_coords[1],
        begin=test_time_range[0],
        end=test_time_range[1],
        step_size=test_step_size,
        observatory_id=test_observatory_id,
        observatory_name=test_observatory_name,
    )
    vis_2 = mock_visibility_class(
        ra=test_coords[0],
        dec=test_coords[1],
        begin=test_time_range[0],
        end=test_time_range[1],
        step_size=test_step_size,
        observatory_id=test_observatory_id_2,
        observatory_name=test_observatory_name_2,
    )

    vis_1.compute()
    vis_2.compute()

    n_samples = len(vis_1.inconstraint)
    inconstraint = np.zeros(n_samples, dtype=bool)
    inconstraint[0] = True

    vis_1.inconstraint = inconstraint
    vis_2.inconstraint = inconstraint.copy()

    return vis_1, vis_2


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
def constraints_from_fixture(constraint_json: str) -> list[AllConstraint]:
    """Fixture that provides constraints loaded from JSON."""
    result = constraints_from_json(constraint_json)
    # Ensure it's always a list for this fixture
    return result if isinstance(result, list) else [result]


# AllConstraint combinations for testing logical operators


@pytest.fixture
def always_satisfied_earth_constraint() -> EarthLimbConstraint:
    """Fixture for an EarthLimbConstraint that is always satisfied (min_angle=0)."""
    return EarthLimbConstraint(min_angle=0)


@pytest.fixture
def sun_constraint_45() -> SunAngleConstraint:
    """Fixture for a SunAngleConstraint with min_angle=45."""
    return SunAngleConstraint(min_angle=45)


@pytest.fixture
def earth_constraint_33() -> EarthLimbConstraint:
    """Fixture for an EarthLimbConstraint with min_angle=33."""
    return EarthLimbConstraint(min_angle=33)


@pytest.fixture
def or_always_satisfied(always_satisfied_earth_constraint: EarthLimbConstraint) -> ConstraintABC:
    """Fixture for an OR constraint of two always-satisfied constraints."""
    return always_satisfied_earth_constraint | always_satisfied_earth_constraint


@pytest.fixture
def and_always_satisfied(always_satisfied_earth_constraint: EarthLimbConstraint) -> ConstraintABC:
    """Fixture for an AND constraint of two always-satisfied constraints."""
    return always_satisfied_earth_constraint & always_satisfied_earth_constraint


@pytest.fixture
def xor_always_satisfied(always_satisfied_earth_constraint: EarthLimbConstraint) -> ConstraintABC:
    """Fixture for an XOR constraint of two always-satisfied constraints."""
    return always_satisfied_earth_constraint ^ always_satisfied_earth_constraint


@pytest.fixture
def xor_sun_earth(
    sun_constraint_45: SunAngleConstraint, earth_constraint_33: EarthLimbConstraint
) -> ConstraintABC:
    """Fixture for an XOR constraint combining SUN and EARTH constraints."""
    return sun_constraint_45 ^ earth_constraint_33


@pytest.fixture
def or_sun_earth(
    sun_constraint_45: SunAngleConstraint, earth_constraint_33: EarthLimbConstraint
) -> ConstraintABC:
    """Fixture for an OR constraint combining SUN and EARTH constraints."""
    return sun_constraint_45 | earth_constraint_33


@pytest.fixture
def not_or_sun_earth(or_sun_earth: ConstraintABC) -> ConstraintABC:
    """Fixture for a NOT constraint wrapping OR(SUN, EARTH)."""
    return ~or_sun_earth


@pytest.fixture
def and_or_sun_earth(or_sun_earth: ConstraintABC) -> ConstraintABC:
    """Fixture for an AND constraint of two OR(SUN, EARTH) constraints."""
    return or_sun_earth & or_sun_earth


# Catalog test fixtures


@pytest.fixture(autouse=True)
def clear_catalog_cache() -> Generator[None, None, None]:
    """Automatically clear catalog cache before each test."""
    from across.tools.visibility.catalogs import cache_clear

    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def test_star_coord() -> SkyCoord:
    """Fixture for a test star coordinate (Sirius position)."""
    return SkyCoord(ra="06h45m08.9s", dec="-16d42m58.0s", frame="icrs")


@pytest.fixture
def mock_vizier_table() -> Table:
    """Fixture for a mock Vizier table with test star data."""

    mock_table = Table()
    mock_table["_RA.icrs"] = [101.28]
    mock_table["_DE.icrs"] = [-16.72]
    mock_table["Vmag"] = [-1.46]
    return mock_table


@pytest.fixture
def mock_vizier_table_alternate_columns() -> Table:
    """Fixture for a mock Vizier table with alternate column names."""
    mock_table = Table()
    mock_table["RAJ2000"] = [101.28]
    mock_table["DEJ2000"] = [-16.72]
    mock_table["Vmag"] = [-1.46]
    return mock_table


@pytest.fixture
def mock_bright_stars() -> list[tuple[SkyCoord, float]]:
    """Fixture providing a small set of mock bright stars for testing.

    Returns a list of (SkyCoord, magnitude) tuples representing
    common bright stars used in tests. Prevents internet access
    during constraint testing.
    """
    return [
        # Sirius (brightest star, used in many tests)
        (SkyCoord(ra="06h45m08.9s", dec="-16d42m58.0s", frame="icrs"), -1.46),
        # Canopus
        (SkyCoord(ra="06h23m57.1s", dec="-52d41m44.4s", frame="icrs"), -0.74),
        # Arcturus
        (SkyCoord(ra="14h15m39.7s", dec="+19d10m56.7s", frame="icrs"), -0.05),
        # Vega
        (SkyCoord(ra="18h36m56.3s", dec="+38d47m01.3s", frame="icrs"), 0.03),
        # Capella
        (SkyCoord(ra="05h16m41.4s", dec="+45d59m52.8s", frame="icrs"), 0.08),
    ]


@pytest.fixture
def mock_vizier_instance(mock_vizier_table: Table) -> MagicMock:
    """Fixture providing a pre-configured mock Vizier instance."""
    mock_instance = MagicMock()
    mock_instance.query_constraints.return_value = [mock_vizier_table]
    return mock_instance


@pytest.fixture
def mock_vizier_patch(mock_vizier_instance: MagicMock) -> Generator[MagicMock, None, None]:
    """Fixture providing a patched Vizier context manager."""
    with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
        mock_vizier.return_value = mock_vizier_instance
        yield mock_vizier


@pytest.fixture
def fallback_bright_stars() -> list[tuple[SkyCoord, float]]:
    """Fixture providing the fallback bright stars list."""
    from across.tools.visibility.catalogs import _get_fallback_bright_stars

    return _get_fallback_bright_stars()


@pytest.fixture
def mock_cache_dir_patch(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture providing a patched cache directory."""
    with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def mock_vizier_magnitude_filtering() -> Generator[MagicMock, None, None]:
    """Fixture providing a Vizier mock that returns different results based on magnitude limit."""

    def return_stars_by_mag(**kwargs: Any) -> list[Table]:
        vmag = kwargs.get("Vmag")
        # Parse magnitude limit
        if vmag is not None and "<3" in vmag:
            # Fewer stars for mag < 3
            mock_table = Table()
            mock_table["_RA.icrs"] = [101.28]
            mock_table["_DE.icrs"] = [-16.72]
            mock_table["Vmag"] = [-1.46]
            return [mock_table]
        else:
            # More stars for mag < 6 (return multiple rows)
            extended_table = Table()
            extended_table["_RA.icrs"] = [101.28, 200.0, 250.0]
            extended_table["_DE.icrs"] = [-16.72, 30.0, -40.0]
            extended_table["Vmag"] = [-1.46, 3.5, 5.0]
            return [extended_table]

    with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
        mock_instance = MagicMock()
        mock_instance.query_constraints.side_effect = return_stars_by_mag
        mock_vizier.return_value = mock_instance
        yield mock_vizier
