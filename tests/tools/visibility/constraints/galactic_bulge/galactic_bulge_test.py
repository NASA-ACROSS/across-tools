import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.galactic_bulge import GalacticBulgeConstraint


class TestGalacticBulgeConstraintAttributes:
    """Test suite for GalacticBulgeConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = GalacticBulgeConstraint()
        assert constraint.short_name == "Galactic Bulge"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = GalacticBulgeConstraint()
        assert constraint.name.value == "Galactic Bulge Avoidance"


class TestGalacticBulgeConstraintInitialization:
    """Test suite for GalacticBulgeConstraint initialization."""

    def test_constraint_initialization_default_min_separation(self) -> None:
        """Test constraint initialization with default min_separation."""
        constraint = GalacticBulgeConstraint()
        assert constraint.min_separation == 10.0

    def test_constraint_initialization_custom_min_separation(self) -> None:
        """Test constraint initialization with custom min_separation."""
        constraint = GalacticBulgeConstraint(min_separation=20.0)
        assert constraint.min_separation == 20.0

    def test_constraint_initialization_zero_min_separation_raises_error(self) -> None:
        """Test constraint initialization with zero min_separation raises error."""
        with pytest.raises(ValueError):
            GalacticBulgeConstraint(min_separation=0.0)  # Must be > 0

    def test_constraint_initialization_negative_min_separation_raises_error(self) -> None:
        """Test constraint initialization with negative min_separation raises error."""
        with pytest.raises(ValueError):
            GalacticBulgeConstraint(min_separation=-5.0)  # Must be > 0


class TestGalacticBulgeConstraintCall:
    """Test suite for GalacticBulgeConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        galactic_bulge_constraint: GalacticBulgeConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = galactic_bulge_constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        galactic_bulge_constraint: GalacticBulgeConstraint,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = galactic_bulge_constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_no_violation_when_far_from_bulge(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        far_from_bulge_coord: SkyCoord,
        galactic_bulge_constraint: GalacticBulgeConstraint,
    ) -> None:
        """Test constraint returns False when coordinate is far from galactic bulge."""
        result = galactic_bulge_constraint(begin_time_array, ground_ephemeris, far_from_bulge_coord)
        # Should not be constrained (far from bulge)
        assert not result

    def test_constraint_violation_when_close_to_bulge(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        near_bulge_coord: SkyCoord,
        galactic_bulge_constraint: GalacticBulgeConstraint,
    ) -> None:
        """Test constraint returns True when coordinate is close to galactic bulge."""
        result = galactic_bulge_constraint(begin_time_array, ground_ephemeris, near_bulge_coord)
        # Should be constrained (within 10° of bulge)
        assert result

    def test_constraint_at_bulge_center(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        bulge_center_coord: SkyCoord,
        galactic_bulge_constraint: GalacticBulgeConstraint,
    ) -> None:
        """Test constraint with coordinate at galactic bulge center."""
        result = galactic_bulge_constraint(begin_time_array, ground_ephemeris, bulge_center_coord)
        # Should always be constrained (0° separation < 10°)
        assert result

    def test_constraint_with_small_min_separation_at_bulge_center(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        bulge_center_coord: SkyCoord,
    ) -> None:
        """Test constraint with small min_separation at bulge center."""
        constraint_small = GalacticBulgeConstraint(min_separation=1.0)  # 0° < 1°, constrained
        result_small = constraint_small(begin_time_array, ground_ephemeris, bulge_center_coord)

        assert result_small

    def test_constraint_with_large_min_separation_at_bulge_center(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        bulge_center_coord: SkyCoord,
    ) -> None:
        """Test constraint with large min_separation at bulge center."""
        constraint_large = GalacticBulgeConstraint(min_separation=30.0)  # 0° < 30°, constrained
        result_large = constraint_large(begin_time_array, ground_ephemeris, bulge_center_coord)

        assert result_large
