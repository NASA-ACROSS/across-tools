from datetime import datetime, timedelta

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.alt_az import AltAzConstraint


class TestAltAzConstraintAttributes:
    """Test suite for AltAzConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        assert constraint.short_name == "AltAz"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        assert constraint.name.value == "Altitude/Azimuth Avoidance"


class TestAltAzConstraintInitialization:
    """Test suite for AltAzConstraint initialization."""

    def test_constraint_initialization_all_none_polygon(self) -> None:
        """Test constraint initialization with all None - polygon."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        assert constraint.polygon is None

    def test_constraint_initialization_all_none_altitude_min(self) -> None:
        """Test constraint initialization with all None - altitude_min."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        assert constraint.altitude_min is None

    def test_constraint_initialization_all_none_altitude_max(self) -> None:
        """Test constraint initialization with all None - altitude_max."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        assert constraint.altitude_max is None

    def test_constraint_initialization_with_values_polygon(self) -> None:
        """Test constraint initialization with values - polygon."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon, altitude_min=15.0, altitude_max=75.0)
        assert constraint.polygon == polygon

    def test_constraint_initialization_with_values_altitude_min(self) -> None:
        """Test constraint initialization with values - altitude_min."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon, altitude_min=15.0, altitude_max=75.0)
        assert constraint.altitude_min == 15.0

    def test_constraint_initialization_with_values_altitude_max(self) -> None:
        """Test constraint initialization with values - altitude_max."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon, altitude_min=15.0, altitude_max=75.0)
        assert constraint.altitude_max == 75.0

    def test_constraint_initialization_with_values_azimuth_min(self) -> None:
        """Test constraint initialization with values - azimuth_min."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon, azimuth_min=15.0, azimuth_max=75.0)
        assert constraint.azimuth_min == 15.0

    def test_constraint_initialization_with_values_azimuth_max(self) -> None:
        """Test constraint initialization with values - azimuth_max."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon, azimuth_min=15.0, azimuth_max=75.0)
        assert constraint.azimuth_max == 75.0


class TestAltAzConstraintCalculation:
    """Test suite for the AltAzConstraint class."""

    def test_constraint_with_min_altitude_returns_ndarray(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with minimum altitude returns ndarray."""

        constraint = AltAzConstraint(polygon=None, altitude_min=30.0, altitude_max=None)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)
        assert isinstance(result, np.ndarray)

    def test_constraint_with_min_altitude_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, az_zero_alt_forty_five_sky_coord: SkyCoord
    ) -> None:
        """Test constraint with minimum altitude returns bool dtype."""
        constraint = AltAzConstraint(polygon=None, altitude_min=30.0, altitude_max=None)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)
        assert result.dtype == bool

    def test_constraint_no_restrictions_returns_all_false(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, az_zero_alt_forty_five_sky_coord: SkyCoord
    ) -> None:
        """Test constraint with no restrictions returns all False."""
        constraint = AltAzConstraint(
            polygon=None,
        )
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)
        assert not result.any()

    def test_constraint_with_altitude_max_above_pointing_returns_false(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with maximum altitude returns False."""
        constraint = AltAzConstraint(polygon=None, altitude_min=None, altitude_max=60.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.False_

    def test_constraint_with_altitude_max_less_than_pointing_returns_false(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with maximum altitude returns False."""
        constraint = AltAzConstraint(polygon=None, altitude_min=None, altitude_max=30.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.True_

    def test_constraint_with_azimuth_max_greater_than_pointing_returns_false(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with maximum azimuth returns False."""
        constraint = AltAzConstraint(polygon=None, azimuth_min=None, azimuth_max=60.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.False_

    def test_constraint_with_azimuth_max_below_pointing_returns_true(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with maximum azimuth returns True."""
        constraint = AltAzConstraint(polygon=None, azimuth_min=None, azimuth_max=30.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.True_

    def test_constraint_with_azimuth_min_greater_than_pointing_returns_true(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with minimum azimuth returns True."""
        constraint = AltAzConstraint(polygon=None, azimuth_max=None, azimuth_min=60.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.True_

    def test_constraint_with_azimuth_min_less_than_pointing_returns_false(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with minimum azimuth returns False."""
        constraint = AltAzConstraint(polygon=None, azimuth_max=None, azimuth_min=30.0)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.False_

    def test_constraint_multiple_times_returns_correct_length(
        self,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
        ephemeris_begin: datetime,
        ephemeris_step_size: int,
    ) -> None:
        """Test constraint with multiple times returns correct length."""
        times = Time([ephemeris_begin, ephemeris_begin + timedelta(seconds=ephemeris_step_size)], scale="utc")
        constraint = AltAzConstraint(polygon=None, altitude_min=30.0, altitude_max=None)
        result = constraint(times, ground_ephemeris, az_zero_alt_forty_five_sky_coord)
        assert len(result) == len(times)

    def test_constraint_with_polygon_overlapping_pointing_returns_true(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_eight_alt_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with polygon overlapping pointing returns True."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(
            polygon=polygon,
        )
        result = constraint(begin_time_array, ground_ephemeris, az_eight_alt_five_sky_coord)

        assert result[0] is np.True_

    def test_constraint_with_pointing_outside_polygon_returns_false(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        az_zero_alt_forty_five_sky_coord: SkyCoord,
    ) -> None:
        """Test constraint with polygon overlapping pointing returns True."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        constraint = AltAzConstraint(polygon=polygon)
        result = constraint(begin_time_array, ground_ephemeris, az_zero_alt_forty_five_sky_coord)

        assert result[0] is np.False_
