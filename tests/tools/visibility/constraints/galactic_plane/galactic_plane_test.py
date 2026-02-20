import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.galactic_plane import GalacticPlaneConstraint


class TestGalacticPlaneConstraintAttributes:
    """Test suite for GalacticPlaneConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = GalacticPlaneConstraint()
        assert constraint.short_name == "Galactic Plane"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = GalacticPlaneConstraint()
        assert constraint.name.value == "Galactic Plane Avoidance"


class TestGalacticPlaneConstraintInitialization:
    """Test suite for GalacticPlaneConstraint initialization."""

    def test_constraint_initialization_default_min_latitude(self) -> None:
        """Test constraint initialization with default min_latitude."""
        constraint = GalacticPlaneConstraint()
        assert constraint.min_latitude == 10.0

    def test_constraint_initialization_custom_min_latitude(self) -> None:
        """Test constraint initialization with custom min_latitude."""
        constraint = GalacticPlaneConstraint(min_latitude=20.0)
        assert constraint.min_latitude == 20.0

    def test_constraint_initialization_negative_min_latitude_raises_error(self) -> None:
        """Test constraint initialization with negative min_latitude raises error."""
        with pytest.raises(ValueError):
            GalacticPlaneConstraint(min_latitude=-5.0)  # Below minimum of 0

    def test_constraint_initialization_too_large_min_latitude_raises_error(self) -> None:
        """Test constraint initialization with too-large min_latitude raises error."""
        with pytest.raises(ValueError):
            GalacticPlaneConstraint(min_latitude=95.0)  # Above maximum of 90


class TestGalacticPlaneConstraintCall:
    """Test suite for GalacticPlaneConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_high_galactic_latitude(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        high_galactic_latitude_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test constraint returns False when galactic latitude is above min_latitude."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, high_galactic_latitude_coord)
        # Should not be constrained (latitude = 60° > 10°)
        assert not result

    def test_constraint_violation_when_low_galactic_latitude(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        low_galactic_latitude_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test constraint returns True when galactic latitude is below min_latitude."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, low_galactic_latitude_coord)
        # Should be constrained (latitude = 5° < 10°)
        assert result

    def test_constraint_at_galactic_equator(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        galactic_equator_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test constraint with coordinate at galactic equator (latitude = 0°)."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, galactic_equator_coord)
        # Should always be constrained (0° < 10°)
        assert result

    def test_constraint_with_low_min_latitude_threshold(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        low_galactic_latitude_coord: SkyCoord,
    ) -> None:
        """Test low min_latitude threshold does not constrain coordinate."""
        constraint_low = GalacticPlaneConstraint(min_latitude=2.0)  # 5° > 2°, not constrained

        result_low = constraint_low(begin_time_array, ground_ephemeris, low_galactic_latitude_coord)

        # Lower threshold should not constrain this coordinate
        assert not result_low  # 5° > 2°, not constrained

    def test_constraint_with_high_min_latitude_threshold(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        low_galactic_latitude_coord: SkyCoord,
    ) -> None:
        """Test high min_latitude threshold constrains coordinate."""
        constraint_high = GalacticPlaneConstraint(min_latitude=15.0)  # 5° < 15°, constrained
        result_high = constraint_high(begin_time_array, ground_ephemeris, low_galactic_latitude_coord)

        assert result_high  # 5° < 15°, constrained

    def test_constraint_south_galactic_pole(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        south_galactic_pole_coord: SkyCoord,
        galactic_plane_constraint: GalacticPlaneConstraint,
    ) -> None:
        """Test constraint with coordinate at south galactic pole."""
        result = galactic_plane_constraint(begin_time_array, ground_ephemeris, south_galactic_pole_coord)
        # Should never be constrained (90° > 10°)
        assert not result
