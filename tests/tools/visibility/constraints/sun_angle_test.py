import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
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
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert isinstance(result, np.ndarray)

    def test_sun_angle_constraint_call_returns_bool_dtype(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns boolean dtype."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result.dtype == np.bool_

    def test_sun_angle_constraint_call_returns_correct_length(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns array of correct length."""
        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == len(test_tle_ephemeris.timestamp)

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
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that the constraint correctly identifies coordinates not in the constraint."""
        sun_coord = test_tle_ephemeris.sun[0]
        opposite_sun = sun_coord.directional_offset_by(0 * u.deg, 160 * u.deg)

        sky_coord = SkyCoord(ra=opposite_sun.ra, dec=opposite_sun.dec)

        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.all(result == np.False_)

    def test_sun_angle_constraint_in_constraint(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that the constraint correctly identifies coordinates in the constraint."""
        inside_coord = test_tle_ephemeris.sun[0]
        sky_coord = SkyCoord(ra=inside_coord.ra, dec=inside_coord.dec)

        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.all(result == np.True_)

    def test_sun_angle_constraint_edge_of_constraint(
        self, sun_angle_constraint: SunAngleConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that the constraint correctly identifies coordinates at the edge of the constraint."""

        # Create a SkyCoord that is at the edge of the Sun angle constraint
        sky_coord = SkyCoord(
            test_tle_ephemeris.sun[2].ra, test_tle_ephemeris.sun[2].dec, unit="deg", frame="icrs"
        ).directional_offset_by(180 * u.deg, sun_angle_constraint.min_angle * u.deg)

        result = sun_angle_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.array_equal(
            result,
            np.array(
                [
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ]
            ),
        )
