import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints import SunAngleConstraint


class TestSunAngleConstraint:
    """Test suite for the SunAngleConstraint class."""

    def test_sun_angle_constraint_short_name(self, sun_angle_constraint: SunAngleConstraint) -> None:
        """Test that SunAngleConstraint has correct short_name."""
        assert sun_angle_constraint.short_name == "Sun"

    def test_sun_angle_constraint_name_value(self, sun_angle_constraint: SunAngleConstraint) -> None:
        """Test that SunAngleConstraint has correct name value."""
        assert sun_angle_constraint.name.value == "Sun Angle"

    def test_sun_angle_constraint_min_angle(self, sun_angle_constraint: SunAngleConstraint) -> None:
        """Test that SunAngleConstraint has correct min_angle."""
        assert sun_angle_constraint.min_angle == 45.0

    def test_sun_angle_constraint_max_angle(self, sun_angle_constraint: SunAngleConstraint) -> None:
        """Test that SunAngleConstraint has correct max_angle."""
        assert sun_angle_constraint.max_angle == 170.0

    def test_sun_angle_constraint_call_returns_ndarray(
        self, sun_constraint_result: npt.NDArray[np.bool_]
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        assert isinstance(sun_constraint_result, np.ndarray)

    def test_sun_angle_constraint_call_returns_bool_dtype(
        self, sun_constraint_result: npt.NDArray[np.bool_]
    ) -> None:
        """Test that __call__ method returns boolean dtype."""
        assert sun_constraint_result.dtype == np.bool_

    def test_sun_angle_constraint_call_returns_correct_length(
        self, sun_constraint_result: npt.NDArray[np.bool_], test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns array of correct length."""
        assert len(sun_constraint_result) == len(test_tle_ephemeris.timestamp)

    def test_sun_angle_constraint_call_time_subset_length(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns correct length for time subset."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp[1:3], ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == 2

    def test_sun_angle_constraint_call_time_outside_ephemeris_bounds(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method raises AssertionError for time outside ephemeris bounds."""
        with pytest.raises(AssertionError):
            sun_angle_constraint(
                time=Time(["2023-10-01T00:00:00", "2023-10-01T00:01:00"]),
                ephemeris=test_tle_ephemeris,
                coordinate=sky_coord,
            )

    def test_sun_angle_constraint_not_in_constraint(
        self,
        sun_angle_constraint: SunAngleConstraint,
        sun_outside_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that the constraint correctly identifies coordinates not in the constraint."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=sun_outside_constraint_coord,
        )
        assert np.all(result == np.False_)

    def test_sun_angle_constraint_in_constraint(
        self,
        sun_angle_constraint: SunAngleConstraint,
        sun_inside_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that the constraint correctly identifies coordinates in the constraint."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=sun_inside_constraint_coord,
        )
        assert np.all(result == np.True_)

    def test_sun_angle_constraint_edge_of_constraint(
        self,
        sun_angle_constraint: SunAngleConstraint,
        sun_edge_constraint_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that the constraint correctly identifies coordinates at the edge of the constraint."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp,
            ephemeris=test_tle_ephemeris,
            coordinate=sun_edge_constraint_coord,
        )

        # Assert that as we're on the edge of a constraint, the 5 computed
        # values should contain True and False
        assert np.any(result) and np.any(~result)
