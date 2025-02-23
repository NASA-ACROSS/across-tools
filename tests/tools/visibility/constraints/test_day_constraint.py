import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.core.enums import TwilightType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.day import DayConstraint


class TestDayConstraint:
    """Test the DayConstraint class."""

    def test_astronomical_constraint_returns_true(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that astronomical twilight constraint returns True."""
        constraint = DayConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_nautical_constraint_returns_true(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that nautical twilight constraint returns True."""
        constraint = DayConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_civil_constraint_returns_true(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that civil twilight constraint returns True."""
        constraint = DayConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_sunrise_constraint_returns_true(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that sunrise constraint returns True."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_horizon_dip_constraint_returns_true(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint with horizon dip returns True."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE, horizon_dip=True)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_array_time_returns_numpy_array(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that using array of times returns numpy array."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert isinstance(result, np.ndarray)

    def test_array_time_returns_correct_length(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that array result has correct length."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert len(result) == 6
