import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints import EarthLimbConstraint


class TestEarthLimbConstraint:
    """Test suite for the EarthLimbConstraint class."""

    def test_earth_limb_constraint_short_name(self, earth_limb_constraint: EarthLimbConstraint) -> None:
        """Test that EarthLimbConstraint has correct short_name."""
        assert earth_limb_constraint.short_name == "Earth"

    def test_earth_limb_constraint_name_value(self, earth_limb_constraint: EarthLimbConstraint) -> None:
        """Test that EarthLimbConstraint has correct name value."""
        assert earth_limb_constraint.name.value == "Earth Limb"

    def test_earth_limb_constraint_min_angle(self, earth_limb_constraint: EarthLimbConstraint) -> None:
        """Test that EarthLimbConstraint has correct min_angle."""
        assert earth_limb_constraint.min_angle == 33.0

    def test_earth_limb_constraint_max_angle(self, earth_limb_constraint: EarthLimbConstraint) -> None:
        """Test that EarthLimbConstraint has correct max_angle."""
        assert earth_limb_constraint.max_angle == 170.0

    def test_earth_limb_constraint_call_returns_ndarray(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert isinstance(result, np.ndarray)

    def test_earth_limb_constraint_call_returns_bool_dtype(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns boolean dtype."""
        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result.dtype == np.bool_

    def test_earth_limb_constraint_call_returns_correct_length(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns array with correct length."""
        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_earth_limb_constraint_call_time_subset_length(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method with time subset returns correct length."""
        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp[1:3], ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == 2

    def test_earth_limb_constraint_call_time_outside_ephemeris_bounds(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method raises AssertionError for time outside ephemeris bounds."""
        with pytest.raises(AssertionError):
            earth_limb_constraint(
                time=Time(["2023-10-01T00:00:00", "2023-10-01T00:01:00"]),
                ephemeris=test_tle_ephemeris,
                coordinate=sky_coord,
            )

    def test_earth_limb_constraint_not_in_constraint(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that constraint correctly identifies coordinates not in the constraint."""
        sky_coord = SkyCoord(
            test_tle_ephemeris.earth[2].ra, test_tle_ephemeris.earth[2].dec, unit="deg", frame="icrs"
        ).directional_offset_by(
            0 * u.deg,
            10 * u.deg + earth_limb_constraint.min_angle * u.deg + test_tle_ephemeris.earth_radius_angle,
        )

        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.all(result == np.False_)

    def test_earth_limb_constraint_in_constraint(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that constraint correctly identifies coordinates in the constraint."""
        inside_coord = SkyCoord(test_tle_ephemeris.earth[0].ra, test_tle_ephemeris.earth[0].dec)

        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=inside_coord
        )
        assert np.all(result == np.True_)

    def test_earth_limb_constraint_edge_contains_true(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that edge of constraint contains True values."""
        sky_coord = SkyCoord(
            test_tle_ephemeris.earth[2].ra, test_tle_ephemeris.earth[2].dec, unit="deg", frame="icrs"
        ).directional_offset_by(
            0 * u.deg, earth_limb_constraint.min_angle * u.deg + test_tle_ephemeris.earth_radius_angle
        )

        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert True in result

    def test_earth_limb_constraint_edge_contains_false(
        self, earth_limb_constraint: EarthLimbConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that edge of constraint contains False values."""
        sky_coord = SkyCoord(
            test_tle_ephemeris.earth[2].ra, test_tle_ephemeris.earth[2].dec, unit="deg", frame="icrs"
        ).directional_offset_by(
            0 * u.deg, earth_limb_constraint.min_angle * u.deg + test_tle_ephemeris.earth_radius_angle
        )

        result = earth_limb_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert False in result
