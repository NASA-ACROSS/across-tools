import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

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

    def test_daytime_constraint_sunset_validation_has_two_transitions(
        self, mauna_kea_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that daytime constraint identifies exactly 2 transitions (sunrise/sunset)."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        results = constraint(mauna_kea_ephemeris.timestamp, mauna_kea_ephemeris, dummy_coord)
        transitions = np.where(np.diff(results.astype(int)) != 0)[0] + 1
        assert len(transitions) == 2

    def test_daytime_constraint_sunset_validation_sunrise_time_accuracy(
        self, mauna_kea_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that daytime constraint sunrise time is close to independent calculation."""
        from astropy.coordinates import AltAz

        ephem = mauna_kea_ephemeris
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        results = constraint(ephem.timestamp, ephem, dummy_coord)
        transitions = np.where(np.diff(results.astype(int)) != 0)[0] + 1
        sunrise_idx = transitions[0]
        constraint_sunrise = ephem.timestamp[sunrise_idx]

        location = ephem.earth_location
        twilight_altitude = -6 * u.deg
        sun_altitudes = ephem.sun.transform_to(AltAz(obstime=ephem.timestamp, location=location)).alt
        below_horizon = sun_altitudes < twilight_altitude
        sunrise_mask = ~below_horizon[:-1] & below_horizon[1:]
        independent_sunrise_idx = np.where(sunrise_mask)[0]

        if len(independent_sunrise_idx) > 0:
            independent_sunrise = ephem.timestamp[independent_sunrise_idx[0]]
            time_diff_sunrise = abs((constraint_sunrise - independent_sunrise).sec) / 60
            assert time_diff_sunrise < 15

    def test_daytime_constraint_sunset_validation_sunset_time_accuracy(
        self, mauna_kea_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that daytime constraint sunset time is close to independent calculation."""
        from astropy.coordinates import AltAz

        ephem = mauna_kea_ephemeris
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        results = constraint(ephem.timestamp, ephem, dummy_coord)
        transitions = np.where(np.diff(results.astype(int)) != 0)[0] + 1
        sunset_idx = transitions[1]
        constraint_sunset = ephem.timestamp[sunset_idx]

        location = ephem.earth_location
        twilight_altitude = -6 * u.deg
        sun_altitudes = ephem.sun.transform_to(AltAz(obstime=ephem.timestamp, location=location)).alt
        below_horizon = sun_altitudes < twilight_altitude
        sunset_mask = below_horizon[:-1] & ~below_horizon[1:]
        independent_sunset_idx = np.where(sunset_mask)[0]

        if len(independent_sunset_idx) > 0:
            independent_sunset = ephem.timestamp[independent_sunset_idx[0]]
            time_diff_sunset = abs((constraint_sunset - independent_sunset).sec) / 60
            assert time_diff_sunset < 15

    def test_daytime_constraint_sunset_validation_daytime_result_is_boolean(
        self, mauna_kea_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that daytime constraint result at midday is boolean."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(mauna_kea_ephemeris.timestamp, mauna_kea_ephemeris, dummy_coord)
        midday_idx = len(mauna_kea_ephemeris.timestamp) // 2
        assert isinstance(result[midday_idx], (bool, np.bool_))

    def test_daytime_constraint_civil_twilight_type(self) -> None:
        """Test that DaytimeConstraint can be created with CIVIL twilight type."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        assert constraint.twilight_type == TwilightType.CIVIL

    def test_daytime_constraint_nautical_twilight_type(self) -> None:
        """Test that DaytimeConstraint can be created with NAUTICAL twilight type."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        assert constraint.twilight_type == TwilightType.NAUTICAL

    def test_daytime_constraint_astronomical_twilight_type(self) -> None:
        """Test that DaytimeConstraint can be created with ASTRONOMICAL twilight type."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        assert constraint.twilight_type == TwilightType.ASTRONOMICAL

    def test_daytime_constraint_sunset_twilight_type(self) -> None:
        """Test that DaytimeConstraint can be created with SUNSET twilight type."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        assert constraint.twilight_type == TwilightType.SUNSET

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

    def test_daytime_constraint_civil_twilight_ground_based_returns_ndarray(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that civil twilight ground-based result is ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_civil_twilight_ground_based_returns_bool_dtype(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that civil twilight ground-based result is bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_nautical_twilight_ground_based_returns_ndarray(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that nautical twilight ground-based result is ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_nautical_twilight_ground_based_returns_bool_dtype(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that nautical twilight ground-based result is bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_astronomical_twilight_ground_based_returns_ndarray(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that astronomical twilight ground-based result is ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_astronomical_twilight_ground_based_returns_bool_dtype(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that astronomical twilight ground-based result is bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_sunset_twilight_ground_based_returns_ndarray(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that sunset twilight ground-based result is ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_sunset_twilight_ground_based_returns_bool_dtype(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that sunset twilight ground-based result is bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        result = constraint(ground_ephemeris.timestamp, ground_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_ground_based_all_twilights_same_length(
        self, ground_ephemeris: GroundEphemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that all twilight type results have same length."""
        civil_result = DaytimeConstraint(twilight_type=TwilightType.CIVIL)(
            ground_ephemeris.timestamp, ground_ephemeris, dummy_coord
        )
        nautical_result = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)(
            ground_ephemeris.timestamp, ground_ephemeris, dummy_coord
        )
        astronomical_result = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)(
            ground_ephemeris.timestamp, ground_ephemeris, dummy_coord
        )
        sunset_result = DaytimeConstraint(twilight_type=TwilightType.SUNSET)(
            ground_ephemeris.timestamp, ground_ephemeris, dummy_coord
        )
        assert len(civil_result) == len(nautical_result) == len(astronomical_result) == len(sunset_result)

    def test_daytime_constraint_space_based_civil_twilight_returns_ndarray(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based civil twilight constraint returns ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_space_based_civil_twilight_returns_bool_dtype(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based civil twilight constraint returns bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_space_based_civil_twilight_correct_length(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based civil twilight constraint returns correct length."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_daytime_constraint_space_based_nautical_twilight_returns_ndarray(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based nautical twilight constraint returns ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_space_based_nautical_twilight_returns_bool_dtype(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based nautical twilight constraint returns bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_space_based_nautical_twilight_correct_length(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based nautical twilight constraint returns correct length."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_daytime_constraint_space_based_astronomical_twilight_returns_ndarray(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based astronomical twilight constraint returns ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_space_based_astronomical_twilight_returns_bool_dtype(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based astronomical twilight constraint returns bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_space_based_astronomical_twilight_correct_length(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based astronomical twilight constraint returns correct length."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_daytime_constraint_space_based_sunset_twilight_returns_ndarray(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based sunset twilight constraint returns ndarray."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert isinstance(result, np.ndarray)

    def test_daytime_constraint_space_based_sunset_twilight_returns_bool_dtype(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based sunset twilight constraint returns bool dtype."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert result.dtype == np.bool_

    def test_daytime_constraint_space_based_sunset_twilight_correct_length(
        self, test_tle_ephemeris: Ephemeris, dummy_coord: SkyCoord
    ) -> None:
        """Test that space-based sunset twilight constraint returns correct length."""
        constraint = DaytimeConstraint(twilight_type=TwilightType.SUNSET)
        result = constraint(test_tle_ephemeris.timestamp, test_tle_ephemeris, dummy_coord)
        assert len(result) == len(test_tle_ephemeris.timestamp)
