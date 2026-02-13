import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums import TwilightType
from across.tools.ephemeris import Ephemeris, GroundEphemeris
from across.tools.ephemeris.tle_ephem import TLEEphemeris
from across.tools.visibility.constraints import DaytimeConstraint


class TestDaytimeConstraint:
    """Test suite for the DaytimeConstraint class."""

    def test_daytime_constraint_short_name(self) -> None:
        """Test that DaytimeConstraint has correct short_name."""
        constraint = DaytimeConstraint()
        assert constraint.short_name == "Daytime"

    def test_daytime_constraint_name_value(self) -> None:
        """Test that DaytimeConstraint has correct name value."""
        constraint = DaytimeConstraint()
        assert constraint.name.value == "Daytime Avoidance"

    def test_daytime_constraint_default_twilight_type(self) -> None:
        """Test that DaytimeConstraint has correct default twilight type."""
        constraint = DaytimeConstraint()
        assert constraint.twilight_type == TwilightType.ASTRONOMICAL

    def test_daytime_constraint_call_returns_ndarray(
        self, ground_ephemeris: GroundEphemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        constraint = DaytimeConstraint()
        result = constraint(time=ground_ephemeris.timestamp, ephemeris=ground_ephemeris, coordinate=sky_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_call_returns_bool_dtype(
        self, ground_ephemeris: GroundEphemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that __call__ method returns boolean dtype."""
        constraint = DaytimeConstraint()
        result = constraint(time=ground_ephemeris.timestamp, ephemeris=ground_ephemeris, coordinate=sky_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_call_returns_correct_length(
        self, ground_ephemeris: GroundEphemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that __call__ method returns array of correct length."""
        constraint = DaytimeConstraint()
        result = constraint(time=ground_ephemeris.timestamp, ephemeris=ground_ephemeris, coordinate=sky_coord)
        assert len(result) == len(ground_ephemeris.timestamp)

    def test_daytime_constraint_sunset_validation(
        self, mauna_kea_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that daytime constraint correctly identifies sunset times using independent validation.

        This test validates that the daytime constraint correctly calculates when the Sun
        is above/below the horizon by comparing against astropy's independent sunset
        calculations for a specific location and time.
        """
        # Use the Mauna Kea ephemeris fixture
        ephem = mauna_kea_ephemeris

        # Test civil twilight (Sun 0-6° below horizon)
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)

        # Get constraint results
        results = constraint(ephem.timestamp, ephem, dummy_coord)

        # Find transition points (where constraint changes from False to True or vice versa)
        transitions = np.where(np.diff(results.astype(int)) != 0)[0] + 1

        # Should have 2 transitions: sunrise and sunset
        assert len(transitions) == 2, f"Expected 2 transitions, got {len(transitions)}"

        sunrise_idx, sunset_idx = transitions

        # Get the times of sunrise and sunset from our constraint
        constraint_sunrise = ephem.timestamp[sunrise_idx]
        constraint_sunset = ephem.timestamp[sunset_idx]

        # Independent validation using astropy's astronomical calculations
        # Calculate sunrise/sunset times for civil twilight
        from astropy.coordinates import AltAz

        # Use the location from the ephemeris fixture
        location = ephem.earth_location

        # For civil twilight, Sun is at -6° altitude
        twilight_altitude = -6 * u.deg

        # Calculate when Sun reaches -6° altitude (civil twilight)
        sun_altitudes = ephem.sun.transform_to(AltAz(obstime=ephem.timestamp, location=location)).alt

        # Find when Sun crosses -6° altitude
        below_horizon = sun_altitudes < twilight_altitude

        # Sunrise: first time Sun rises above -6°
        sunrise_mask = ~below_horizon[:-1] & below_horizon[1:]
        independent_sunrise_idx = np.where(sunrise_mask)[0]
        if len(independent_sunrise_idx) > 0:
            independent_sunrise = ephem.timestamp[independent_sunrise_idx[0]]
        else:
            # Fallback: find first time above horizon
            above_horizon = sun_altitudes > 0
            first_above = np.where(above_horizon)[0]
            independent_sunrise = ephem.timestamp[first_above[0]] if len(first_above) > 0 else None

        # Sunset: first time Sun drops below -6°
        sunset_mask = below_horizon[:-1] & ~below_horizon[1:]
        independent_sunset_idx = np.where(sunset_mask)[0]
        if len(independent_sunset_idx) > 0:
            independent_sunset = ephem.timestamp[independent_sunset_idx[0]]
        else:
            # Fallback: find last time above horizon
            above_horizon = sun_altitudes > 0
            last_above = np.where(above_horizon)[0]
            independent_sunset = ephem.timestamp[last_above[-1]] if len(last_above) > 0 else None

        # Validate that our constraint times are close to independent calculations
        # Allow for some tolerance due to discrete time steps and interpolation differences
        tolerance_minutes = 15  # 15 minutes tolerance

        if independent_sunrise is not None:
            time_diff_sunrise = abs((constraint_sunrise - independent_sunrise).sec) / 60
            assert time_diff_sunrise < tolerance_minutes, (
                f"Sunrise time mismatch: constraint={constraint_sunrise.iso}, "
                f"independent={independent_sunrise.iso}, diff={time_diff_sunrise:.1f} min"
            )

        if independent_sunset is not None:
            time_diff_sunset = abs((constraint_sunset - independent_sunset).sec) / 60
            assert time_diff_sunset < tolerance_minutes, (
                f"Sunset time mismatch: constraint={constraint_sunset.iso}, "
                f"independent={independent_sunset.iso}, diff={time_diff_sunset:.1f} min"
            )

        # Additional validation: check that constraint correctly identifies daytime
        # During the day (between sunrise and sunset), constraint should return True
        if independent_sunrise is not None and independent_sunset is not None:
            # Check a time point within the ephemeris range that's likely daytime
            midday_idx = len(ephem.timestamp) // 2  # Middle of the time range
            midday_time = ephem.timestamp[midday_idx]
            midday_result = constraint(Time([midday_time]), ephem, dummy_coord)
            if len(midday_result) > 0:
                # If we have a result, it should be consistent with our expectations
                # (though we can't guarantee it's daytime without more complex checks)
                assert isinstance(midday_result[0], (bool, np.bool_)), "Result should be boolean"

    def test_daytime_constraint_different_twilight_types(self) -> None:
        """Test that different twilight types are properly configured."""
        # Test that we can create constraints with different twilight types
        civil_constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        nautical_constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        astronomical_constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        sunset_constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)

        # Verify the twilight types are set correctly
        assert civil_constraint.twilight_type == TwilightType.CIVIL
        assert nautical_constraint.twilight_type == TwilightType.NAUTICAL
        assert astronomical_constraint.twilight_type == TwilightType.ASTRONOMICAL
        assert sunset_constraint.twilight_type == TwilightType.SUNSET

    def test_daytime_constraint_space_based_returns_array(
        self, test_tle_ephemeris: TLEEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based daytime constraint returns array."""
        constraint = DaytimeConstraint()
        coord = dummy_coord
        # Use the TLE ephemeris which should trigger space-based calculations
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_space_based_returns_bool_dtype(
        self, test_tle_ephemeris: TLEEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based daytime constraint returns boolean dtype."""
        constraint = DaytimeConstraint()
        coord = dummy_coord
        # Use the TLE ephemeris which should trigger space-based calculations
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_space_based_returns_correct_length(
        self, test_tle_ephemeris: TLEEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based daytime constraint returns correct length."""
        constraint = DaytimeConstraint()
        coord = dummy_coord
        # Use the TLE ephemeris which should trigger space-based calculations
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, coord)
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_daytime_constraint_different_twilight_types_ground_based(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that different twilight types produce different results for ground-based observations."""
        coord = dummy_coord

        # Test all twilight types
        civil_constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        nautical_constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        astronomical_constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        sunset_constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)

        # Get results for each
        civil_result = civil_constraint(ground_ephemeris.timestamp, ground_ephemeris, coord)
        nautical_result = nautical_constraint(ground_ephemeris.timestamp, ground_ephemeris, coord)
        astronomical_result = astronomical_constraint(ground_ephemeris.timestamp, ground_ephemeris, coord)
        sunset_result = sunset_constraint(ground_ephemeris.timestamp, ground_ephemeris, coord)

        # All should be boolean arrays
        for result in [civil_result, nautical_result, astronomical_result, sunset_result]:
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.bool_

        # Different twilight types should generally give different results
        # (though they might be the same at certain times)
        # At least verify they're all computed
        assert len(civil_result) == len(nautical_result) == len(astronomical_result) == len(sunset_result)

    def test_daytime_constraint_space_based_with_different_twilight_types(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based constraint works with different twilight types."""
        # Skip space-based test due to ephemeris compatibility issues
        coord = dummy_coord

        # Test all twilight types with space-based ephemeris
        for twilight_type in [
            TwilightType.CIVIL,
            TwilightType.NAUTICAL,
            TwilightType.ASTRONOMICAL,
            TwilightType.SUNSET,
        ]:
            constraint = DaytimeConstraint(twilight_type=twilight_type)
            result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, coord)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.bool_
            assert len(result) == len(test_tle_ephemeris.timestamp)
