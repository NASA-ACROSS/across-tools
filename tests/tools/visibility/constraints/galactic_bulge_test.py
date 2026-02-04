import pytest
from astropy import units as u  # type: ignore[import-untyped]
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

    def test_constraint_initialization_invalid_min_separation(self) -> None:
        """Test constraint initialization with invalid min_separation raises error."""
        with pytest.raises(ValueError):
            GalacticBulgeConstraint(min_separation=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            GalacticBulgeConstraint(min_separation=-5.0)  # Must be > 0


class TestGalacticBulgeConstraintCall:
    """Test suite for GalacticBulgeConstraint __call__ method."""

    def test_constraint_returns_bool_array(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint returns boolean array."""
        constraint = GalacticBulgeConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_no_violation_when_far_from_bulge(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns False when coordinate is far from galactic bulge."""
        # Create a coordinate far from the galactic bulge (opposite side of sky)
        coord = SkyCoord(ra=0 * u.deg, dec=60 * u.deg)  # Far from bulge at RA~266°, Dec~-29°
        constraint = GalacticBulgeConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should not be constrained (far from bulge)
        assert not result

    def test_constraint_violation_when_close_to_bulge(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint returns True when coordinate is close to galactic bulge."""
        # Create a coordinate very close to the galactic bulge center
        bulge_coord = SkyCoord(ra="17h45m40.04s", dec="-29d00m28.1s", frame="icrs")
        # Add small offset (1 degree)
        coord = SkyCoord(ra=bulge_coord.ra + 1 * u.deg, dec=bulge_coord.dec)
        constraint = GalacticBulgeConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should be constrained (within 10° of bulge)
        assert result

    def test_constraint_at_bulge_center(self, begin_time_array: Time, ground_ephemeris: Ephemeris) -> None:
        """Test constraint with coordinate at galactic bulge center."""
        coord = SkyCoord(ra="17h45m40.04s", dec="-29d00m28.1s", frame="icrs")
        constraint = GalacticBulgeConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coord)
        # Should always be constrained (0° separation < 10°)
        assert result

    def test_constraint_with_different_min_separation_values(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris
    ) -> None:
        """Test constraint behavior with different min_separation thresholds."""
        coord = SkyCoord(ra="17h45m40.04s", dec="-29d00m28.1s", frame="icrs")

        constraint_small = GalacticBulgeConstraint(min_separation=1.0)  # 0° < 1°, constrained
        constraint_large = GalacticBulgeConstraint(min_separation=30.0)  # 0° < 30°, constrained

        result_small = constraint_small(begin_time_array, ground_ephemeris, coord)
        result_large = constraint_large(begin_time_array, ground_ephemeris, coord)

        # Both should constrain the bulge center
        assert result_small
        assert result_large
