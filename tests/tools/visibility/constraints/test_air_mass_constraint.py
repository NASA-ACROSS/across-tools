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
        return AirMassConstraint(min=1.0, max=2.0)

    def test_init_min(self, constraint: AirMassConstraint) -> None:
        """Test initialization of min."""
        assert constraint.min == 1.0

    def test_init_max(self, constraint: AirMassConstraint) -> None:
        """Test initialization of max."""
        assert constraint.max == 2.0

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
        constraint = AirMassConstraint(min=1.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_with_only_min_return_dtype(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return dtype with only minimum limit."""
        constraint = AirMassConstraint(min=1.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool

    def test_call_with_only_max_return_type(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return type with only maximum limit."""
        constraint = AirMassConstraint(max=2.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert isinstance(result, np.ndarray)

    def test_call_with_only_max_return_dtype(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test return dtype with only maximum limit."""
        constraint = AirMassConstraint(max=2.0)
        time = keck_ground_ephemeris.timestamp
        coord = SkyCoord(ra=45 * u.deg, dec=45 * u.deg)
        result = constraint(time, keck_ground_ephemeris, coord)
        assert result.dtype == bool
