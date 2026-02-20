import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints import MoonAngleConstraint


class TestMoonAngleConstraint:
    """Test suite for the MoonAngleConstraint class."""

    def test_moon_angle_constraint_short_name(self, moon_angle_constraint: MoonAngleConstraint) -> None:
        """Test that MoonAngleConstraint has correct short_name."""
        assert moon_angle_constraint.short_name == "Moon"

    def test_moon_angle_constraint_name_value(self, moon_angle_constraint: MoonAngleConstraint) -> None:
        """Test that MoonAngleConstraint has correct name value."""
        assert moon_angle_constraint.name.value == "Moon Angle"

    def test_moon_angle_constraint_min_angle(self, moon_angle_constraint: MoonAngleConstraint) -> None:
        """Test that MoonAngleConstraint has correct min_angle."""
        assert moon_angle_constraint.min_angle == 21.0

    def test_moon_angle_constraint_max_angle(self, moon_angle_constraint: MoonAngleConstraint) -> None:
        """Test that MoonAngleConstraint has correct max_angle."""
        assert moon_angle_constraint.max_angle == 170.0

    def test_moon_angle_constraint_call_returns_ndarray(
        self, moon_constraint_result: npt.NDArray[np.bool_]
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        assert isinstance(moon_constraint_result, np.ndarray)

    def test_moon_angle_constraint_call_returns_bool_dtype(
        self, moon_constraint_result: npt.NDArray[np.bool_]
    ) -> None:
        """Test that __call__ method returns boolean dtype."""
        assert moon_constraint_result.dtype == np.bool_

    def test_moon_angle_constraint_call_returns_correct_length(
        self, moon_constraint_result: npt.NDArray[np.bool_], test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns array with correct length."""
        assert len(moon_constraint_result) == len(test_tle_ephemeris.timestamp)

    def test_moon_angle_constraint_call_time_subset_length(
        self, moon_angle_constraint: MoonAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method with time subset returns correct length."""
        result = moon_angle_constraint(
            time=test_tle_ephemeris.timestamp[1:3], ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == 2

    def test_moon_angle_constraint_call_time_outside_ephemeris_bounds_raises_error(
        self, moon_angle_constraint: MoonAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method raises AssertionError for time outside ephemeris bounds."""
        with pytest.raises(AssertionError):
            moon_angle_constraint(
                time=Time(["2023-10-01T00:00:00", "2023-10-01T00:01:00"]),
                ephemeris=test_tle_ephemeris,
                coordinate=sky_coord,
            )

    def test_moon_angle_constraint_not_in_constraint_all_false(
        self,
        moon_angle_constraint: MoonAngleConstraint,
        moon_outside_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that coordinates outside constraint return all False."""
        result = moon_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=moon_outside_constraint_coord,
        )
        assert np.all(result == np.False_)

    def test_moon_angle_constraint_in_constraint_all_true(
        self,
        moon_angle_constraint: MoonAngleConstraint,
        moon_inside_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that coordinates inside constraint return all True."""
        result = moon_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=moon_inside_constraint_coord,
        )
        assert np.all(result == np.True_)

    def test_moon_angle_constraint_edge_of_constraint_expected_values(
        self,
        moon_angle_constraint: MoonAngleConstraint,
        moon_edge_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that coordinates at edge of constraint return expected boolean array."""
        result = moon_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=moon_edge_constraint_coord,
        )
        # Assert that as we're on the edge of a constraint, the 5 computed
        # values should contain True and False
        assert np.any(result) and np.any(~result)
