import pytest
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.galactic_plane import GalacticPlaneConstraint


class TestGalacticPlaneConstraintAttributes:
    """Test suite for GalacticPlaneConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = GalacticPlaneConstraint()
        assert constraint.short_name == "Galactic Plane"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = GalacticPlaneConstraint()
        assert constraint.name.value == "Galactic Plane Avoidance"


class TestGalacticPlaneConstraintInitialization:
    """Test suite for GalacticPlaneConstraint initialization."""

    def test_constraint_initialization_default_min_latitude(self) -> None:
        """Test constraint initialization with default min_latitude."""
        constraint = GalacticPlaneConstraint()
        assert constraint.min_latitude == 10.0

    def test_constraint_initialization_custom_min_latitude(self) -> None:
        """Test constraint initialization with custom min_latitude."""
        constraint = GalacticPlaneConstraint(min_latitude=20.0)
        assert constraint.min_latitude == 20.0

    def test_constraint_initialization_invalid_min_latitude(self) -> None:
        """Test constraint initialization with invalid min_latitude raises error."""
        with pytest.raises(ValueError):
            GalacticPlaneConstraint(min_latitude=-5.0)  # Below minimum of 0

        with pytest.raises(ValueError):
            GalacticPlaneConstraint(min_latitude=95.0)  # Above maximum of 90


class TestGalacticPlaneConstraintCall:
    """Test suite for GalacticPlaneConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns array-like result."""
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns boolean dtype."""
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_high_galactic_latitude(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns False when galactic latitude is above min_latitude."""
        # Create a coordinate at high galactic latitude (north galactic pole)
        coord = SkyCoord(l=0 * u.deg, b=60 * u.deg, frame="galactic")  # 60° galactic latitude
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should not be constrained (latitude = 60° > 10°)
        assert not result

    def test_constraint_violation_when_low_galactic_latitude(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns True when galactic latitude is below min_latitude."""
        # Create a coordinate near the galactic plane
        coord = SkyCoord(l=0 * u.deg, b=5 * u.deg, frame="galactic")  # 5° galactic latitude
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should be constrained (latitude = 5° < 10°)
        assert result

    def test_constraint_at_galactic_equator(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint with coordinate at galactic equator (latitude = 0°)."""
        coord = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")  # On galactic equator
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should always be constrained (0° < 10°)
        assert result

    def test_constraint_with_different_min_latitude_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint behavior with different min_latitude thresholds."""
        coord = SkyCoord(l=0 * u.deg, b=5 * u.deg, frame="galactic")  # 5° galactic latitude

        constraint_low = GalacticPlaneConstraint(min_latitude=2.0)  # 5° > 2°, not constrained
        constraint_high = GalacticPlaneConstraint(min_latitude=15.0)  # 5° < 15°, constrained

        result_low = constraint_low(begin_time_array, ground_ephemeris, coord)
        result_high = constraint_high(begin_time_array, ground_ephemeris, coord)

        # Lower threshold should not constrain this coordinate
        assert not result_low  # 5° > 2°, not constrained
        assert result_high  # 5° < 15°, constrained

    def test_constraint_south_galactic_pole(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint with coordinate at south galactic pole."""
        coord = SkyCoord(l=0 * u.deg, b=-90 * u.deg, frame="galactic")  # South galactic pole
        constraint = GalacticPlaneConstraint(min_latitude=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should never be constrained (90° > 10°)
        assert not result
