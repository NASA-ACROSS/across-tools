import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.ecliptic_latitude import EclipticLatitudeConstraint


class TestEclipticLatitudeConstraintAttributes:
    """Test suite for EclipticLatitudeConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.short_name == "Ecliptic Latitude"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.name.value == "Ecliptic Latitude"


class TestEclipticLatitudeConstraintInitialization:
    """Test suite for EclipticLatitudeConstraint initialization."""

    def test_constraint_initialization_default_min_latitude(self) -> None:
        """Test constraint initialization with default min_latitude."""
        constraint = EclipticLatitudeConstraint()
        assert constraint.min_latitude == 15.0

    def test_constraint_initialization_custom_min_latitude(self) -> None:
        """Test constraint initialization with custom min_latitude."""
        constraint = EclipticLatitudeConstraint(min_latitude=20.0)
        assert constraint.min_latitude == 20.0

    def test_constraint_initialization_negative_min_latitude_raises_error(self) -> None:
        """Test constraint initialization with negative min_latitude raises error."""
        with pytest.raises(ValueError):
            EclipticLatitudeConstraint(min_latitude=-5.0)  # Below minimum of 0

    def test_constraint_initialization_too_large_min_latitude_raises_error(self) -> None:
        """Test constraint initialization with too-large min_latitude raises error."""
        with pytest.raises(ValueError):
            EclipticLatitudeConstraint(min_latitude=95.0)  # Above maximum of 90


class TestEclipticLatitudeConstraintCall:
    """Test suite for EclipticLatitudeConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_latitude_above_min(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        high_latitude_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test constraint returns False when ecliptic latitude is above min_latitude."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, high_latitude_coord)
        assert not result.any()

    def test_constraint_violation_when_latitude_below_min(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        ecliptic_equator_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test constraint returns True when ecliptic latitude is below min_latitude."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, ecliptic_equator_coord)
        assert result

    def test_constraint_with_low_min_latitude_threshold(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        ten_degree_latitude_coord: SkyCoord,
    ) -> None:
        """Test low min_latitude threshold does not constrain coordinate."""
        constraint_low = EclipticLatitudeConstraint(min_latitude=5.0)  # 10° > 5°, not constrained

        result_low = constraint_low(begin_time_array, ground_ephemeris, ten_degree_latitude_coord)

        assert not result_low.any()  # 10° > 5°, not constrained

    def test_constraint_with_high_min_latitude_threshold(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        ten_degree_latitude_coord: SkyCoord,
    ) -> None:
        """Test high min_latitude threshold constrains coordinate."""
        constraint_high = EclipticLatitudeConstraint(min_latitude=20.0)  # 10° < 20°, constrained
        result_high = constraint_high(begin_time_array, ground_ephemeris, ten_degree_latitude_coord)

        assert result_high.any()  # 10° < 20°, constrained

    def test_constraint_equator_coordinate(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        ecliptic_equator_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test constraint with coordinate at ecliptic equator (latitude = 0°)."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, ecliptic_equator_coord)
        assert result

    def test_constraint_pole_coordinate(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        ecliptic_pole_coord: SkyCoord,
        ecliptic_constraint: EclipticLatitudeConstraint,
    ) -> None:
        """Test constraint with coordinate at ecliptic pole (latitude = 90°)."""
        result = ecliptic_constraint(begin_time_array, ground_ephemeris, ecliptic_pole_coord)
        assert not result.any()
