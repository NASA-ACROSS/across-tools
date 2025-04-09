import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.air_mass import AirMassConstraint


class TestAirMassConstraint:
    """Test the AirMassConstraint class."""

    @pytest.fixture
    def constraint(self) -> AirMassConstraint:
        """Create a basic AirMassConstraint instance."""
        return AirMassConstraint(airmass_min=1.0, airmass_max=2.0)

    def test_init_airmass_min(self, constraint: AirMassConstraint) -> None:
        """Test initialization of airmass_min."""
        assert constraint.airmass_min == 1.0

    def test_init_airmass_max(self, constraint: AirMassConstraint) -> None:
        """Test initialization of airmass_max."""
        assert constraint.airmass_max == 2.0

    def test_init_name(self, constraint: AirMassConstraint) -> None:
        """Test initialization of name."""
        assert constraint.name == "Airmass"

    def test_init_short_name(self, constraint: AirMassConstraint) -> None:
        """Test initialization of short_name."""
        assert constraint.short_name == ConstraintType.AIR_MASS

    def test_call_return_type(self, constraint: AirMassConstraint, keck_ground_ephemeris: Ephemeris) -> None:
        """Test the return type of __call__ method."""
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_return_dtype(self, constraint: AirMassConstraint, keck_ground_ephemeris: Ephemeris) -> None:
        """Test the return dtype of __call__ method."""
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool

    def test_call_with_none_limits_return_type(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return type with None limits."""
        constraint = AirMassConstraint()
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_with_none_limits_return_dtype(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return dtype with None limits."""
        constraint = AirMassConstraint()
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool

    def test_call_with_only_min_return_type(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return type with only minimum limit."""
        constraint = AirMassConstraint(airmass_min=1.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_with_only_min_return_dtype(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return dtype with only minimum limit."""
        constraint = AirMassConstraint(airmass_min=1.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool

    def test_call_with_only_max_return_type(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return type with only maximum limit."""
        constraint = AirMassConstraint(airmass_max=2.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_with_only_max_return_dtype(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return dtype with only maximum limit."""
        constraint = AirMassConstraint(airmass_max=2.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool
