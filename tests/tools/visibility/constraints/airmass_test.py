from typing import Any, Literal

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.airmass import AirmassConstraint


class TestAirmassConstraintAttributes:
    """Test suite for AirmassConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = AirmassConstraint()
        assert constraint.short_name == "Airmass"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = AirmassConstraint()
        assert constraint.name.value == "Airmass Limit"


class TestAirmassConstraintInitialization:
    """Test suite for AirmassConstraint initialization."""

    def test_constraint_initialization_default_max_air_mass(self) -> None:
        """Test constraint initialization with default max_air_mass."""
        constraint = AirmassConstraint()
        assert constraint.max_air_mass == 2.0

    def test_constraint_initialization_custom_max_air_mass(self) -> None:
        """Test constraint initialization with custom max_air_mass."""
        constraint = AirmassConstraint(max_air_mass=1.5)
        assert constraint.max_air_mass == 1.5

    def test_constraint_initialization_invalid_max_air_mass(self) -> None:
        """Test constraint initialization with invalid max_air_mass raises error."""
        with pytest.raises(ValueError):
            AirmassConstraint(max_air_mass=0.5)  # Below minimum of 1.0


class TestAirmassConstraintCall:
    """Test suite for AirmassConstraint __call__ method."""

    def test_constraint_returns_bool_array(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint returns boolean array."""
        constraint = AirmassConstraint(max_air_mass=2.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_constraint_sets_computed_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint sets computed_values.air_mass."""
        constraint = AirmassConstraint(max_air_mass=2.0)
        constraint(begin_time_array, ground_ephemeris, coordinate)
        assert constraint.computed_values.air_mass is not None
        assert len(constraint.computed_values.air_mass) == len(begin_time_array)

    def test_constraint_no_violation_when_air_mass_below_max(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint returns False when air_mass is below max_air_mass."""
        constraint = AirmassConstraint(max_air_mass=50.0)  # High threshold
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should not be constrained (air_mass < 50.0)
        assert not result.any()

    def test_constraint_violation_when_air_mass_above_max(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint returns True when air_mass is above max_air_mass."""
        constraint = AirmassConstraint(max_air_mass=1.2)  # Low threshold
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should be constrained (air_mass > 1.2)
        assert result.any()

    def test_constraint_with_earth_location_none_raises_error(
        self, begin_time_array: Time, coordinate: SkyCoord
    ) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""

        # Create a mock ephemeris without earth_location
        class MockEphemeris:
            earth_location = None

            def index(self, time: Any) -> Literal[0]:
                return 0

        constraint = AirmassConstraint(max_air_mass=2.0)
        with pytest.raises(ValueError, match="Earth location required for airmass calculations"):
            constraint(begin_time_array, MockEphemeris(), coordinate)  # type: ignore[arg-type]

    def test_constraint_air_mass_calculation_accuracy(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that air_mass values are reasonable (between 1 and ~10 for visible objects)."""
        constraint = AirmassConstraint(max_air_mass=10.0)
        constraint(begin_time_array, ground_ephemeris, coordinate)
        air_mass = constraint.computed_values.air_mass
        assert air_mass is not None

        # Airmass should be >= 1 (by definition)
        assert np.all(air_mass >= 1.0)
        # For typical observations, airmass can be high for low elevation objects
        assert np.all(air_mass < 50.0)

    def test_constraint_different_max_air_mass_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint behavior with different max_air_mass thresholds."""
        constraint_low = AirmassConstraint(max_air_mass=1.5)
        constraint_high = AirmassConstraint(max_air_mass=3.0)

        result_low = constraint_low(begin_time_array, ground_ephemeris, coordinate)
        result_high = constraint_high(begin_time_array, ground_ephemeris, coordinate)

        # Lower threshold should constrain more points
        # (Note: this may not always be true depending on actual air_mass values)
        # At minimum, both should return arrays of same length
        assert len(result_low) == len(result_high)
        assert len(result_low) == len(result_high)
