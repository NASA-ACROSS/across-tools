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

    def test_constraint_initialization_default_min_separation(self) -> None:
        """Test constraint initialization with default min_separation."""
        constraint = SolarSystemConstraint()
        assert constraint.min_separation == 10.0

    def test_constraint_initialization_default_bodies(self) -> None:
        """Test constraint initialization with default bodies."""
        constraint = SolarSystemConstraint()
        assert constraint.bodies == [
            SolarSystemObject.MERCURY,
            SolarSystemObject.VENUS,
            SolarSystemObject.MARS,
            SolarSystemObject.JUPITER,
            SolarSystemObject.SATURN,
        ]

    def test_constraint_initialization_custom_min_separation(self) -> None:
        """Test constraint initialization with custom min_separation."""
        constraint = SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])
        assert constraint.min_separation == 20.0

    def test_constraint_initialization_custom_bodies(self) -> None:
        """Test constraint initialization with custom bodies."""
        constraint = SolarSystemConstraint(min_separation=20.0, bodies=["mars", "jupiter"])
        assert constraint.bodies == [SolarSystemObject.MARS, SolarSystemObject.JUPITER]

    def test_constraint_initialization_invalid_min_separation_zero(self) -> None:
        """Test constraint initialization with invalid min_separation (zero) raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=0.0)  # Must be > 0

    def test_constraint_initialization_invalid_min_separation_negative(self) -> None:
        """Test constraint initialization with invalid min_separation (negative) raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=-1.0)  # Must be > 0


class TestSolarSystemConstraintCall:
    """Test suite for SolarSystemConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns array-like result."""
        constraint = SolarSystemConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test that constraint returns boolean dtype."""
        constraint = SolarSystemConstraint(min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_earth_location_required(self, begin_time_array: Time, sky_coord: SkyCoord) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""

        # Create a mock ephemeris without earth_location
        class MockEphemeris:
            earth_location = None

            def index(self, time: Any) -> Literal[0]:
                return 0

        constraint = SolarSystemConstraint(min_separation=10.0)
        with pytest.raises(ValueError, match="Earth location required for Solar System object positions"):
            constraint(begin_time_array, MockEphemeris(), sky_coord)  # type: ignore[arg-type]

    def test_constraint_with_empty_bodies_list(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with empty bodies list returns False."""
        constraint = SolarSystemConstraint(bodies=[])
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Should not be constrained (no bodies to check)
        assert not result

    def test_constraint_with_single_body_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with single body returns array-like result."""
        constraint = SolarSystemConstraint(bodies=["mars"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Should return boolean array
        assert hasattr(result, "dtype")

    def test_constraint_with_single_body_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with single body returns boolean dtype."""
        constraint = SolarSystemConstraint(bodies=["mars"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_with_multiple_bodies_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with multiple bodies returns array-like result."""
        constraint = SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        # Should return boolean array
        assert hasattr(result, "dtype")

    def test_constraint_with_multiple_bodies_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with multiple bodies returns boolean dtype."""
        constraint = SolarSystemConstraint(bodies=["venus", "mars", "jupiter"], min_separation=10.0)
        result = constraint(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_small_separation_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with small separation returns array-like result."""
        constraint_small = SolarSystemConstraint(min_separation=1.0)
        result_small = constraint_small(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result_small, "dtype")

    def test_constraint_small_separation_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with small separation returns boolean dtype."""
        constraint_small = SolarSystemConstraint(min_separation=1.0)
        result_small = constraint_small(begin_time_array, ground_ephemeris, sky_coord)
        assert result_small.dtype == bool

    def test_constraint_large_separation_returns_array_like(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with large separation returns array-like result."""
        constraint_large = SolarSystemConstraint(min_separation=100.0)
        result_large = constraint_large(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result_large, "dtype")

    def test_constraint_large_separation_returns_bool_dtype(
        self, begin_time_array: Time, ground_ephemeris: Ephemeris, sky_coord: SkyCoord
    ) -> None:
        """Test constraint with large separation returns boolean dtype."""
        constraint_large = SolarSystemConstraint(min_separation=100.0)
        result_large = constraint_large(begin_time_array, ground_ephemeris, sky_coord)
        assert result_large.dtype == bool

    def test_constraint_rejects_sun_body(self) -> None:
        """Test constraint rejects sun in bodies list."""
        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["sun"])

    def test_constraint_rejects_moon_body(self) -> None:
        """Test constraint rejects moon in bodies list."""
        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["moon"])

    def test_constraint_rejects_sun_and_moon_bodies(self) -> None:
        """Test constraint rejects sun and moon in bodies list."""
        with pytest.raises(ValidationError):
            SolarSystemConstraint(bodies=["sun", "moon", "venus"])


class TestCalculateBodyMagnitude:
    """Test suite for SolarSystemConstraint._calculate_body_magnitude method."""

    def test_calculate_magnitude_mercury(
        self, body_coord_1au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Mercury."""
        constraint = SolarSystemConstraint()

        # Update ephemeris sun to 180 deg separation (phase_angle=180)
        sun_coord = SkyCoord(ra=180 * u.deg, dec=0 * u.deg, distance=1 * u.AU)
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])

        # Create a new ephemeris instance with the phase angle we want
        class PhaseEphemeris(Ephemeris):
            def __init__(self, sun: SkyCoord) -> None:
                self.sun = sun

            def prepare_data(self) -> None:
                pass

        ephemeris = PhaseEphemeris(sun_array)
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.MERCURY, body_coord_1au, ephemeris, i)
        phase_angle = 180.0
        expected = -1.9 + 0.02 * phase_angle + 3.5e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_venus(self, body_coord_1au: SkyCoord) -> None:
        """Test magnitude calculation for Venus."""
        constraint = SolarSystemConstraint()

        # Mock body_coord at 1 AU, sun at 90 deg separation (phase_angle=90)
        sun_coord = SkyCoord(ra=90 * u.deg, dec=0 * u.deg, distance=1 * u.AU)

        class PhaseEphemeris(Ephemeris):
            def __init__(self, sun: SkyCoord) -> None:
                self.sun = sun

            def prepare_data(self) -> None:
                pass

        # Create SkyCoord array from single coordinate
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
        ephemeris = PhaseEphemeris(sun_array)
        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(SolarSystemObject.VENUS, body_coord_1au, ephemeris, i)
        phase_angle = 90.0
        expected = -4.7 + 0.013 * phase_angle + 4.3e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_mars(
        self, body_coord_1_5au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Mars."""
        constraint = SolarSystemConstraint()

        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(
            SolarSystemObject.MARS, body_coord_1_5au, mock_ephemeris_with_sun, i
        )
        distance_au = 1.5
        expected = -1.52 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_jupiter(
        self, body_coord_5au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Jupiter."""
        constraint = SolarSystemConstraint()

        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(
            SolarSystemObject.JUPITER, body_coord_5au, mock_ephemeris_with_sun, i
        )
        distance_au = 5
        expected = -9.4 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_saturn(
        self, body_coord_9au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Saturn."""
        constraint = SolarSystemConstraint()

        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(
            SolarSystemObject.SATURN, body_coord_9au, mock_ephemeris_with_sun, i
        )
        distance_au = 9
        expected = -8.9 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_uranus(
        self, body_coord_19au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Uranus."""
        constraint = SolarSystemConstraint()

        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(
            SolarSystemObject.URANUS, body_coord_19au, mock_ephemeris_with_sun, i
        )
        distance_au = 19
        expected = 5.5 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_neptune(
        self, body_coord_30au: SkyCoord, mock_ephemeris_with_sun: Ephemeris
    ) -> None:
        """Test magnitude calculation for Neptune."""
        constraint = SolarSystemConstraint()

        i = slice(0, 1)

        result = constraint._calculate_body_magnitude(
            SolarSystemObject.NEPTUNE, body_coord_30au, mock_ephemeris_with_sun, i
        )
        distance_au = 30
        expected = 7.8 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])
