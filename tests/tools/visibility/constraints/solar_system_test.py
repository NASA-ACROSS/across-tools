from typing import Any, Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.core.enums.solar_system_object import SolarSystemObject
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.solar_system import SolarSystemConstraint


class TestSolarSystemConstraintAttributes:
    """Test suite for SolarSystemConstraint attributes."""

    def test_constraint_short_name(self) -> None:
        """Test constraint short_name attribute."""
        constraint = SolarSystemConstraint()
        assert constraint.short_name == "Solar System"

    def test_constraint_name_value(self) -> None:
        """Test constraint name.value attribute."""
        constraint = SolarSystemConstraint()
        assert constraint.name.value == "Solar System Object Avoidance"


class TestSolarSystemConstraintInitialization:
    """Test suite for SolarSystemConstraint initialization."""

    def test_constraint_initialization_default_values(self) -> None:
        """Test constraint initialization with default values."""
        constraint = SolarSystemConstraint()
        assert constraint.min_separation == 10.0
        assert constraint.bodies == [
            SolarSystemObject.MERCURY,
            SolarSystemObject.VENUS,
            SolarSystemObject.MARS,
            SolarSystemObject.JUPITER,
            SolarSystemObject.SATURN,
        ]

    def test_constraint_initialization_custom_values(self) -> None:
        """Test constraint initialization with custom values."""
        constraint = SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])
        assert constraint.min_separation == 20.0
        assert constraint.bodies == [SolarSystemObject.MARS, SolarSystemObject.JUPITER]

    def test_constraint_initialization_invalid_min_separation(self) -> None:
        """Test constraint initialization with invalid min_separation raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=-1.0)  # Must be > 0


class TestSolarSystemConstraintCall:
    """Test suite for SolarSystemConstraint __call__ method."""

    def test_constraint_returns_bool_array(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test that constraint returns boolean array."""
        constraint = SolarSystemConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_earth_location_required(self, begin_time_array: Time, coordinate: SkyCoord) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""

        # Create a mock ephemeris without earth_location
        class MockEphemeris:
            earth_location = None

            def index(self, time: Any) -> Literal[0]:
                return 0

        constraint = SolarSystemConstraint(min_separation=10.0)
        with pytest.raises(ValueError, match="Earth location required for Solar System object positions"):
            constraint(begin_time_array, MockEphemeris(), coordinate)  # type: ignore[arg-type]

    def test_constraint_with_empty_bodies_list(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with empty bodies list returns False."""
        constraint = SolarSystemConstraint(bodies=[])
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should not be constrained (no bodies to check)
        assert not result

    def test_constraint_with_single_body(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with single body in list."""
        constraint = SolarSystemConstraint(bodies=["mars"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should return boolean array
        assert hasattr(result, "dtype")
        assert result.dtype == bool

    def test_constraint_with_multiple_bodies(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, coordinate: SkyCoord
    ) -> None:
        """Test constraint with multiple bodies."""
        constraint = SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, coordinate)
        # Should return boolean array
        assert hasattr(result, "dtype")
        assert result.dtype == bool
        constraint_small = SolarSystemConstraint(min_separation=1.0)
        constraint_large = SolarSystemConstraint(min_separation=100.0)

        result_small = constraint_small(begin_time_array, ground_ephemeris, coordinate)
        result_large = constraint_large(begin_time_array, ground_ephemeris, coordinate)

        # Results should be boolean arrays
        for result in [result_small, result_large]:
            assert hasattr(result, "dtype")
            assert result.dtype == bool

    def test_constraint_rejects_sun_and_moon_bodies(self) -> None:
        """Test constraint rejects sun and moon in bodies list."""
        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["sun"])

        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["moon"])

        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["sun", "moon", "venus"])


class TestCalculateBodyMagnitude:
    """Test suite for SolarSystemConstraint._calculate_body_magnitude method."""

    def test_calculate_magnitude_mercury(self) -> None:
        """Test magnitude calculation for Mercury."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 1 AU, sun at 180 deg separation (phase_angle=180)
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)
        sun_coord = SkyCoord(ra=180 * u.deg, dec=0 * u.deg, distance=1 * u.AU)

        class MockEphemeris(Ephemeris):
            def __init__(self, sun: SkyCoord) -> None:
                self.sun = sun

            def prepare_data(self) -> None:
                pass

        # Create SkyCoord array from single coordinate
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
        ephemeris = MockEphemeris(sun_array)
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.MERCURY, body_coord, ephemeris, i)
        phase_angle = 180.0
        expected = -1.9 + 0.02 * phase_angle + 3.5e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_venus(self) -> None:
        """Test magnitude calculation for Venus."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 1 AU, sun at 90 deg separation (phase_angle=90)
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)
        sun_coord = SkyCoord(ra=90 * u.deg, dec=0 * u.deg, distance=1 * u.AU)

        class MockEphemeris(Ephemeris):
            def __init__(self, sun: SkyCoord) -> None:
                self.sun = sun

            def prepare_data(self) -> None:
                pass

        # Create SkyCoord array from single coordinate
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
        ephemeris = MockEphemeris(sun_array)
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.VENUS, body_coord, ephemeris, i)
        phase_angle = 90.0
        expected = -4.7 + 0.013 * phase_angle + 4.3e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_mars(self) -> None:
        """Test magnitude calculation for Mars."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 1.5 AU
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1.5 * u.AU)

        class MockEphemeris(Ephemeris):
            def prepare_data(self) -> None:
                pass

        ephemeris = MockEphemeris(begin=Time.now(), end=Time.now())
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.MARS, body_coord, ephemeris, i)
        distance_au = 1.5
        expected = -1.52 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_jupiter(self) -> None:
        """Test magnitude calculation for Jupiter."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 5 AU
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=5 * u.AU)

        class MockEphemeris(Ephemeris):
            def prepare_data(self) -> None:
                pass

        ephemeris = MockEphemeris(begin=Time.now(), end=Time.now())
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.JUPITER, body_coord, ephemeris, i)
        distance_au = 5
        expected = -9.4 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_saturn(self) -> None:
        """Test magnitude calculation for Saturn."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 9 AU
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=9 * u.AU)

        class MockEphemeris(Ephemeris):
            def prepare_data(self) -> None:
                pass

        ephemeris = MockEphemeris(begin=Time.now(), end=Time.now())
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.SATURN, body_coord, ephemeris, i)
        distance_au = 9
        expected = -8.9 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_uranus(self) -> None:
        """Test magnitude calculation for Uranus."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 19 AU
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=19 * u.AU)

        class MockEphemeris(Ephemeris):
            def prepare_data(self) -> None:
                pass

        ephemeris = MockEphemeris(begin=Time.now(), end=Time.now())
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.URANUS, body_coord, ephemeris, i)
        distance_au = 19
        expected = 5.5 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_neptune(self) -> None:
        """Test magnitude calculation for Neptune."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 30 AU
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=30 * u.AU)

        class MockEphemeris(Ephemeris):
            def prepare_data(self) -> None:
                pass

        ephemeris = MockEphemeris(begin=Time.now(), end=Time.now())
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.NEPTUNE, body_coord, ephemeris, i)
        distance_au = 30
        expected = 7.8 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])
