from unittest.mock import MagicMock

import pytest
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

    def test_constraint_initialization_default_min_separation(self) -> None:
        """Test constraint initialization default min_separation."""
        constraint = BrightStarConstraint()
        assert constraint.min_separation == 5.0

    def test_constraint_initialization_default_magnitude_limit(self) -> None:
        """Test constraint initialization default magnitude_limit."""
        constraint = BrightStarConstraint()
        assert constraint.magnitude_limit == 6.0

    def test_constraint_initialization_custom_min_separation(self) -> None:
        """Test constraint initialization custom min_separation."""
        constraint = BrightStarConstraint(min_separation=10.0, magnitude_limit=4.0)
        assert constraint.min_separation == 10.0

    def test_constraint_initialization_custom_magnitude_limit(self) -> None:
        """Test constraint initialization custom magnitude_limit."""
        constraint = BrightStarConstraint(min_separation=10.0, magnitude_limit=4.0)
        assert constraint.magnitude_limit == 4.0

    def test_constraint_initialization_zero_min_separation_raises_error(self) -> None:
        """Test constraint initialization with zero min_separation raises error."""
        with pytest.raises(ValueError):
            BrightStarConstraint(min_separation=0.0)  # Must be > 0

    def test_constraint_initialization_negative_min_separation_raises_error(self) -> None:
        """Test constraint initialization with negative min_separation raises error."""
        with pytest.raises(ValueError):
            BrightStarConstraint(min_separation=-1.0)  # Must be > 0


class TestBrightStarConstraintCall:
    """Test suite for BrightStarConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        mock_get_bright_stars: MagicMock,
        bright_star_constraint: BrightStarConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = bright_star_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        mock_get_bright_stars: MagicMock,
        bright_star_constraint: BrightStarConstraint,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = bright_star_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_far_from_stars(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        far_from_bright_stars_coord: SkyCoord,
    ) -> None:
        """Test constraint returns False when coordinate is far from bright stars."""
        constraint = BrightStarConstraint(min_separation=5.0, magnitude_limit=2.0)
        result = constraint(begin_time_array, ground_ephemeris, far_from_bright_stars_coord)
        # Should not be constrained (far from very bright stars)
        assert not result

    def test_constraint_violation_when_close_to_sirius(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        near_sirius_coord: SkyCoord,
        bright_star_constraint: BrightStarConstraint,
    ) -> None:
        """Test constraint returns True when coordinate is close to Sirius."""
        result = bright_star_constraint(begin_time_array, ground_ephemeris, near_sirius_coord)
        # Should be constrained (within 5° of Sirius)
        assert result

    def test_constraint_at_sirius_position(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        sirius_coord: SkyCoord,
        bright_star_constraint: BrightStarConstraint,
    ) -> None:
        """Test constraint with coordinate at Sirius position."""
        result = bright_star_constraint(begin_time_array, ground_ephemeris, sirius_coord)
        # Should always be constrained (0° separation < 5°)
        assert result

    def test_constraint_with_small_min_separation_at_sirius(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        sirius_coord: SkyCoord,
    ) -> None:
        """Test small min_separation at Sirius still constrains."""
        constraint_small = BrightStarConstraint(min_separation=1.0)  # 0° < 1°, constrained
        result_small = constraint_small(begin_time_array, ground_ephemeris, sirius_coord)

        assert result_small

    def test_constraint_with_large_min_separation_at_sirius(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        sirius_coord: SkyCoord,
    ) -> None:
        """Test large min_separation at Sirius constrains."""
        constraint_large = BrightStarConstraint(min_separation=30.0)  # 0° < 30°, constrained
        result_large = constraint_large(begin_time_array, ground_ephemeris, sirius_coord)
        assert result_large

    def test_constraint_with_multiple_coordinates_result_length(
        self,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        mixed_bright_star_coords: SkyCoord,
        mixed_bright_star_times: Time,
    ) -> None:
        """Test result length matches input length for multiple coordinates."""
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(mixed_bright_star_times, ground_ephemeris, mixed_bright_star_coords)

        assert len(result) == len(mixed_bright_star_times)

    def test_constraint_with_multiple_coordinates_bool_dtype(
        self,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        mixed_bright_star_coords: SkyCoord,
        mixed_bright_star_times: Time,
    ) -> None:
        """Test result dtype is bool for multiple coordinates."""
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(mixed_bright_star_times, ground_ephemeris, mixed_bright_star_coords)

        assert result.dtype == bool

    def test_constraint_with_multiple_coordinates_marks_sirius_point(
        self,
        ground_ephemeris: Ephemeris,
        mock_get_bright_stars: MagicMock,
        mixed_bright_star_coords: SkyCoord,
        mixed_bright_star_times: Time,
    ) -> None:
        """Test Sirius coordinate is constrained in mixed coordinate set."""
        constraint = BrightStarConstraint(min_separation=5.0)
        result = constraint(mixed_bright_star_times, ground_ephemeris, mixed_bright_star_coords)

        assert result[1]  # Second coordinate is at Sirius
