from typing import Any, Literal

import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.solar_system import SolarSystemConstraint


class TestSolarSystemConstraintAttributes:
    """Test suite for SolarSystemConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = SolarSystemConstraint()
        assert constraint.short_name == "Solar System"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = SolarSystemConstraint()
        assert constraint.name.value == "Solar System Object Avoidance"


class TestSolarSystemConstraintInitialization:
    """Test suite for SolarSystemConstraint initialization."""

    def test_constraint_initialization_default_values(self) -> None:
        """Test constraint initialization with default values."""
        constraint = SolarSystemConstraint()
        assert constraint.min_separation == 10.0
        assert constraint.bodies == ["venus", "mars", "jupiter", "saturn"]

    def test_constraint_initialization_custom_values(self) -> None:
        """Test constraint initialization with custom values."""
        constraint = SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])
        assert constraint.min_separation == 20.0
        assert constraint.bodies == ["mars", "jupiter"]

    def test_constraint_initialization_invalid_min_separation(self) -> None:
        """Test constraint initialization with invalid min_separation raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=-1.0)  # Must be > 0


class TestSolarSystemConstraintCall:
    """Test suite for SolarSystemConstraint __call__ method."""

    def test_constraint_returns_bool_array(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint returns boolean array."""
        constraint = SolarSystemConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_earth_location_required(self, begin_time_array: Time, coordinate: SkyCoord) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""

        # Create a mock ephemeris without earth_location
        class MockEphemeris:
            earth_location = None

            def index(self, time: Any) -> Literal[0]:
                return 0

        constraint = SolarSystemConstraint(min_separation=10.0)
        with pytest.raises(ValueError, match="Earth location required for Solar System object positions"):
            constraint(begin_time_array, MockEphemeris(), coordinate)  # type: ignore[arg-type]

    def test_constraint_with_empty_bodies_list(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with empty bodies list returns False."""
        constraint = SolarSystemConstraint(bodies=[])
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should not be constrained (no bodies to check)
        assert not result

    def test_constraint_with_single_body(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with single body in list."""
        constraint = SolarSystemConstraint(bodies=["mars"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should return boolean array
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_with_multiple_bodies(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with multiple bodies."""
        constraint = SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should return boolean array
        assert hasattr(result, "dtype")
        assert result.dtype == bool
        constraint_small = SolarSystemConstraint(min_separation=1.0)
        constraint_large = SolarSystemConstraint(min_separation=100.0)

        result_small = constraint_small(begin_time_array, ground_ephemeris, coordinate)
        result_large = constraint_large(begin_time_array, ground_ephemeris, coordinate)

        # Results should be boolean arrays
        for result in [result_small, result_large]:
            assert hasattr(result, "dtype")
            assert result.dtype == bool

    def test_constraint_handles_invalid_body_names(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint handles invalid body names gracefully."""
        constraint = SolarSystemConstraint(bodies=["invalid_body_name"], min_separation=10.0)
        # Should not raise an exception, just skip invalid bodies
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should return boolean array
        assert hasattr(result, "dtype")
        assert result.dtype == bool
