import pytest
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.ecliptic_latitude import EclipticLatitudeConstraint


class TestEclipticLatitudeConstraintAttributes:
    """Test suite for EclipticLatitudeConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.short_name == "Ecliptic Latitude"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.name.value == "Ecliptic Latitude"


class TestEclipticLatitudeConstraintInitialization:
    """Test suite for EclipticLatitudeConstraint initialization."""

    def test_constraint_initialization_default_min_latitude(self) -> None:
        """Test constraint initialization with default min_latitude."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.min_latitude == 15.0

    def test_constraint_initialization_custom_min_latitude(self) -> None:
        """Test constraint initialization with custom min_latitude."""
        constraint = EclipticLatitudeConstraint(min_latitude=20.0)
        assert constraint.min_latitude == 20.0

    def test_constraint_initialization_invalid_min_latitude(self) -> None:
        """Test constraint initialization with invalid min_latitude raises error."""
        with pytest.raises(ValueError):
            EclipticLatitudeConstraint(min_latitude=-5.0)  # Below minimum of 0

        with pytest.raises(ValueError):
            EclipticLatitudeConstraint(min_latitude=95.0)  # Above maximum of 90


class TestEclipticLatitudeConstraintCall:
    """Test suite for EclipticLatitudeConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns array-like result."""
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns boolean dtype."""
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_latitude_above_min(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns False when ecliptic latitude is above min_latitude."""
        # Create a coordinate at high ecliptic latitude (far from ecliptic plane)
        coord = SkyCoord(ra=90 * u.deg, dec=60 * u.deg)  # High latitude coordinate
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should not be constrained (latitude > 15°)
        assert not result.any()

    def test_constraint_violation_when_latitude_below_min(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns True when ecliptic latitude is below min_latitude."""
        # Create a coordinate on the ecliptic plane (latitude = 0°)
        coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)  # On ecliptic equator
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should be constrained (latitude = 0° < 15°)
        assert result

    def test_constraint_with_different_min_latitude_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint behavior with different min_latitude thresholds."""
        coord = SkyCoord(ra=90 * u.deg, dec=10 * u.deg)  # 10° ecliptic latitude

        constraint_low = EclipticLatitudeConstraint(min_latitude=5.0)  # 10° > 5°, not constrained
        constraint_high = EclipticLatitudeConstraint(min_latitude=20.0)  # 10° < 20°, constrained

        result_low = constraint_low(begin_time_array, ground_ephemeris, coord)
        result_high = constraint_high(begin_time_array, ground_ephemeris, coord)

        # Lower threshold should constrain fewer points
        # High threshold should constrain this coordinate
        assert not result_low.any()  # 10° > 5°, not constrained
        assert result_high.any()  # 10° < 20°, constrained

    def test_constraint_equator_coordinate(self, begin_time_array: Time, ground_ephemeris: Ephemeris) -> None:
        """Test constraint with coordinate at ecliptic equator (latitude = 0°)."""
        coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)  # On ecliptic equator
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should always be constrained (0° < 15°)
        assert result

    def test_constraint_pole_coordinate(self, begin_time_array: Time, ground_ephemeris: Ephemeris) -> None:
        """Test constraint with coordinate at ecliptic pole (latitude = 90°)."""
        coord = SkyCoord(ra=90 * u.deg, dec=90 * u.deg)  # At ecliptic pole
        constraint = EclipticLatitudeConstraint(min_latitude=15.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should never be constrained (90° > 15°)
        assert not result.any()
