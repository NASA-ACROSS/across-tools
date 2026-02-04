import pytest
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.bright_star import BrightStarConstraint


class TestBrightStarConstraintAttributes:
    """Test suite for BrightStarConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = BrightStarConstraint()
        assert constraint.short_name == "Bright Star"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = BrightStarConstraint()
        assert constraint.name.value == "Bright Star Avoidance"


class TestBrightStarConstraintInitialization:
    """Test suite for BrightStarConstraint initialization."""

    def test_constraint_initialization_default_values(self) -> None:
        """Test constraint initialization with default values."""
        constraint = BrightStarConstraint()
        assert constraint.min_separation == 5.0
        assert constraint.magnitude_limit == 6.0

    def test_constraint_initialization_custom_values(self) -> None:
        """Test constraint initialization with custom values."""
        constraint = BrightStarConstraint(min_separation=10.0, magnitude_limit=4.0)
        assert constraint.min_separation == 10.0
        assert constraint.magnitude_limit == 4.0

    def test_constraint_initialization_invalid_min_separation(self) -> None:
        """Test constraint initialization with invalid min_separation raises error."""
        with pytest.raises(ValueError):
            BrightStarConstraint(min_separation=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            BrightStarConstraint(min_separation=-1.0)  # Must be > 0


class TestBrightStarConstraintCall:
    """Test suite for BrightStarConstraint __call__ method."""

    def test_constraint_returns_bool_array(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint returns boolean array."""
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_no_violation_when_far_from_stars(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns False when coordinate is far from bright stars."""
        # Create a coordinate far from bright stars (e.g., near celestial pole)
        coord = SkyCoord(ra=0 * u.deg, dec=80 * u.deg)  # Near north celestial pole
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should not be constrained (far from bright stars)
        assert not result

    def test_constraint_violation_when_close_to_sirius(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns True when coordinate is close to Sirius."""
        # Create a coordinate very close to Sirius
        sirius = SkyCoord(ra="05h16m41.4s", dec="-08d12m05.9s")
        # Add small offset (1 degree)
        coord = SkyCoord(ra=sirius.ra + 1 * u.deg, dec=sirius.dec)
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should be constrained (within 5° of Sirius)
        assert result

    def test_constraint_at_sirius_position(self, begin_time_array: Time, ground_ephemeris: Ephemeris) -> None:
        """Test constraint with coordinate at Sirius position."""
        coord = SkyCoord(ra="05h16m41.4s", dec="-08d12m05.9s")
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should always be constrained (0° separation < 5°)
        assert result

    def test_constraint_with_different_min_separation_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint behavior with different min_separation thresholds."""
        coord = SkyCoord(ra="05h16m41.4s", dec="-08d12m05.9s")  # At Sirius

        constraint_small = BrightStarConstraint(min_separation=1.0)  # 0° < 1°, constrained
        constraint_large = BrightStarConstraint(min_separation=30.0)  # 0° < 30°, constrained

        result_small = constraint_small(begin_time_array, ground_ephemeris, coord)
        result_large = constraint_large(begin_time_array, ground_ephemeris, coord)

        # Both should constrain Sirius
        assert result_small
        assert result_large

    def test_constraint_with_multiple_coordinates(self, ground_ephemeris: Ephemeris) -> None:
        """Test constraint with multiple coordinates including some near bright stars."""
        # Create multiple coordinates
        coords = SkyCoord(
            ra=["00h00m00s", "05h16m41.4s", "10h00m00s"], dec=["00d00m00s", "-08d12m05.9s", "00d00m00s"]
        )

        times = Time(["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"])

        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(times, ground_ephemeris, coords)

        # Should return array with same length as input
        assert len(result) == len(times)
        assert result.dtype == bool
        # At least the Sirius coordinate should be constrained
        assert result[1]  # Second coordinate is at Sirius
