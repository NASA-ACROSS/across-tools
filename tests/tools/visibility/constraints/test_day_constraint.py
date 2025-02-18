import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.core.enums import TwilightType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.day import DayConstraint


class TestDayConstraint:
    """Test the DayConstraint class."""

    def test_day_constraint_astronomical(self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord) -> None:
        """Test the DayConstraint with astronomical twilight."""
        constraint = DayConstraint(twilight_type=TwilightType.ASTRONOMICAL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_day_constraint_nautical(self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord) -> None:
        """Test the DayConstraint with nautical twilight."""
        constraint = DayConstraint(twilight_type=TwilightType.NAUTICAL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_day_constraint_civil(self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord) -> None:
        """Test the DayConstraint with civil twilight."""
        constraint = DayConstraint(twilight_type=TwilightType.CIVIL)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_day_constraint_sunrise(self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord) -> None:
        """Test the DayConstraint with sunrise."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_day_constraint_with_horizon_dip(
        self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test the DayConstraint with horizon dip."""
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE, horizon_dip=True)
        result = constraint(keck_ground_ephemeris.timestamp, keck_ground_ephemeris, sky_coord)
        assert result.all() is np.True_

    def test_day_constraint_array_time(self, keck_ground_ephemeris: Ephemeris, sky_coord: SkyCoord) -> None:
        """Test the DayConstraint with an array of times."""
        times = keck_ground_ephemeris.timestamp
        keck_ground_ephemeris.timestamp = times
        constraint = DayConstraint(twilight_type=TwilightType.SUNRISE)
        result = constraint(times, keck_ground_ephemeris, sky_coord)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
