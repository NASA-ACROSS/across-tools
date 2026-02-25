from typing import Any, Literal, cast

import astropy.units as u  # type: ignore[import-untyped]
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

    def test_constraint_initialization_custom_max_air_mass(
        self, airmass_constraint_custom: AirmassConstraint
    ) -> None:
        """Test constraint initialization with custom max_air_mass."""
        assert airmass_constraint_custom.max_air_mass == 1.5

    def test_constraint_initialization_invalid_max_air_mass(self) -> None:
        """Test constraint initialization with invalid max_air_mass raises error."""
        with pytest.raises(ValueError):
            AirmassConstraint(max_air_mass=0.5)  # Below minimum of 1.0


class TestAirmassConstraintCall:
    """Test suite for AirmassConstraint __call__ method."""

    def test_constraint_returns_ndarray(
        self,
        airmass_constraint: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that constraint returns numpy ndarray."""
        result = airmass_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert isinstance(result, np.ndarray)

    def test_constraint_returns_bool_dtype(
        self,
        airmass_constraint: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = airmass_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_sets_air_mass_values(
        self,
        airmass_constraint: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that constraint sets computed_values.air_mass."""
        airmass_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert airmass_constraint.computed_values.air_mass is not None

    def test_constraint_air_mass_array_length(
        self,
        airmass_constraint: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that air_mass array has correct length."""
        airmass_constraint(begin_time_array, ground_ephemeris, sky_coord)
        air_mass = cast(u.Quantity, airmass_constraint.computed_values.air_mass)
        assert len(air_mass) == len(begin_time_array)

    def test_constraint_no_violation_when_air_mass_below_max(
        self,
        airmass_constraint_high_threshold: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test constraint returns False when air_mass is below max_air_mass."""
        result = airmass_constraint_high_threshold(begin_time_array, ground_ephemeris, sky_coord)
        # Should not be constrained (air_mass < 50.0)
        assert not result.any()

    def test_constraint_violation_when_air_mass_above_max(
        self,
        airmass_constraint_low_threshold: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test constraint returns True when air_mass is above max_air_mass."""
        result = airmass_constraint_low_threshold(begin_time_array, ground_ephemeris, sky_coord)
        # Should be constrained (air_mass > 1.2)
        assert result.any()

    def test_constraint_with_earth_location_none_raises_error(
        self, airmass_constraint: AirmassConstraint, begin_time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""

        # Create a mock ephemeris without earth_location
        class MockEphemeris:
            earth_location = None

            def index(self, time: Any) -> Literal[0]:
                return 0

        with pytest.raises(ValueError, match="Earth location required for airmass calculations"):
            airmass_constraint(begin_time_array, MockEphemeris(), sky_coord)  # type: ignore[arg-type]

    def test_constraint_air_mass_is_calculated(
        self,
        airmass_constraint_high_max: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that air_mass values are calculated."""
        airmass_constraint_high_max(begin_time_array, ground_ephemeris, sky_coord)
        air_mass = airmass_constraint_high_max.computed_values.air_mass
        assert air_mass is not None

    def test_constraint_air_mass_minimum_value(
        self,
        airmass_constraint_high_max: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that airmass values are >= 1.0 by definition."""
        airmass_constraint_high_max(begin_time_array, ground_ephemeris, sky_coord)
        air_mass = cast(u.Quantity, airmass_constraint_high_max.computed_values.air_mass)
        assert np.all(air_mass >= 1.0)

    def test_constraint_air_mass_maximum_reasonable_value(
        self,
        airmass_constraint_high_max: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test that airmass values are reasonable (< 50.0 for typical observations)."""
        airmass_constraint_high_max(begin_time_array, ground_ephemeris, sky_coord)
        air_mass = cast(u.Quantity, airmass_constraint_high_max.computed_values.air_mass)
        assert np.all(air_mass < 50.0)

    def test_constraint_different_max_air_mass_values_same_length(
        self,
        airmass_constraint_custom: AirmassConstraint,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
    ) -> None:
        """Test constraint behavior with different max_air_mass thresholds returns same length."""
        constraint_high = AirmassConstraint(max_air_mass=3.0)

        result_low = airmass_constraint_custom(begin_time_array, ground_ephemeris, sky_coord)
        result_high = constraint_high(begin_time_array, ground_ephemeris, sky_coord)

        # Both should return arrays of same length
        assert len(result_low) == len(result_high)
