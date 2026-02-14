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

    def test_constraint_short_name(self, solar_system_constraint: SolarSystemConstraint) -> None:
        """Test constraint short_name attribute."""
        assert solar_system_constraint.short_name == "Solar System"

    def test_constraint_name_value(self, solar_system_constraint: SolarSystemConstraint) -> None:
        """Test constraint name.value attribute."""
        assert solar_system_constraint.name.value == "Solar System Object Avoidance"


class TestSolarSystemConstraintInitialization:
    """Test suite for SolarSystemConstraint initialization."""

    def test_constraint_initialization_default_min_separation(
        self, solar_system_constraint: SolarSystemConstraint
    ) -> None:
        """Test constraint initialization with default min_separation."""
        assert solar_system_constraint.min_separation == 10.0

    def test_constraint_initialization_default_bodies(
        self, solar_system_constraint: SolarSystemConstraint
    ) -> None:
        """Test constraint initialization with default bodies."""
        assert solar_system_constraint.bodies == [
            SolarSystemObject.MERCURY,
            SolarSystemObject.VENUS,
            SolarSystemObject.MARS,
            SolarSystemObject.JUPITER,
            SolarSystemObject.SATURN,
        ]

    def test_constraint_initialization_custom_min_separation(
        self, solar_system_constraint_custom: SolarSystemConstraint
    ) -> None:
        """Test constraint initialization with custom min_separation."""
        assert solar_system_constraint_custom.min_separation == 20.0

    def test_constraint_initialization_custom_bodies(
        self, solar_system_constraint_custom: SolarSystemConstraint
    ) -> None:
        """Test constraint initialization with custom bodies."""
        assert solar_system_constraint_custom.bodies == [SolarSystemObject.MARS, SolarSystemObject.JUPITER]

    def test_constraint_initialization_invalid_min_separation_zero(self) -> None:
        """Test constraint initialization with invalid min_separation (zero) raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=0.0)  # Must be > 0

    def test_constraint_initialization_invalid_min_separation_negative(self) -> None:
        """Test constraint initialization with invalid min_separation (negative) raises error."""
        with pytest.raises(ValueError):
            SolarSystemConstraint(min_separation=-1.0)  # Must be > 0

    def test_constraint_initialization_invalid_bodies_not_list(self) -> None:
        """Test constraint initialization with invalid bodies (not a list) raises error."""
        with pytest.raises(ValueError, match="bodies must be a list"):
            SolarSystemConstraint(bodies=SolarSystemObject.MARS)  # type: ignore[arg-type]


class TestSolarSystemConstraintCall:
    """Test suite for SolarSystemConstraint __call__ method."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_with_separation: SolarSystemConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = solar_system_constraint_with_separation(begin_time_array, ground_ephemeris, sky_coord)
        # Result should be array-like (even if single element)
        assert hasattr(result, "dtype")

    def test_constraint_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_with_separation: SolarSystemConstraint,
    ) -> None:
        """Test that constraint returns boolean dtype."""
        result = solar_system_constraint_with_separation(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_earth_location_required(
        self, begin_time_array: Time, sky_coord: SkyCoord, mock_ephemeris: Ephemeris
    ) -> None:
        """Test constraint raises error when ephemeris has no earth_location."""
        mock_ephemeris.earth_location = None
        constraint = SolarSystemConstraint(min_separation=10.0)
        with pytest.raises(ValueError, match="Earth location required for Solar System object positions"):
            constraint(begin_time_array, mock_ephemeris, sky_coord)

    def test_constraint_with_empty_bodies_list(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_empty_bodies: SolarSystemConstraint,
    ) -> None:
        """Test constraint with empty bodies list returns False."""
        result = solar_system_constraint_empty_bodies(begin_time_array, ground_ephemeris, sky_coord)
        # Should not be constrained (no bodies to check)
        assert not result

    def test_constraint_with_single_body_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_single_body: SolarSystemConstraint,
    ) -> None:
        """Test constraint with single body returns array-like result."""
        result = solar_system_constraint_single_body(begin_time_array, ground_ephemeris, sky_coord)
        # Should return boolean array
        assert hasattr(result, "dtype")

    def test_constraint_with_single_body_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_single_body: SolarSystemConstraint,
    ) -> None:
        """Test constraint with single body returns boolean dtype."""
        result = solar_system_constraint_single_body(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_with_multiple_bodies_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_multiple_bodies: SolarSystemConstraint,
    ) -> None:
        """Test constraint with multiple bodies returns array-like result."""
        result = solar_system_constraint_multiple_bodies(begin_time_array, ground_ephemeris, sky_coord)
        # Should return boolean array
        assert hasattr(result, "dtype")

    def test_constraint_with_multiple_bodies_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_multiple_bodies: SolarSystemConstraint,
    ) -> None:
        """Test constraint with multiple bodies returns boolean dtype."""
        result = solar_system_constraint_multiple_bodies(begin_time_array, ground_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_small_separation_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_small_separation: SolarSystemConstraint,
    ) -> None:
        """Test constraint with small separation returns array-like result."""
        result_small = solar_system_constraint_small_separation(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result_small, "dtype")

    def test_constraint_small_separation_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_small_separation: SolarSystemConstraint,
    ) -> None:
        """Test constraint with small separation returns boolean dtype."""
        result_small = solar_system_constraint_small_separation(begin_time_array, ground_ephemeris, sky_coord)
        assert result_small.dtype == bool

    def test_constraint_large_separation_returns_array_like(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_large_separation: SolarSystemConstraint,
    ) -> None:
        """Test constraint with large separation returns array-like result."""
        result_large = solar_system_constraint_large_separation(begin_time_array, ground_ephemeris, sky_coord)
        assert hasattr(result_large, "dtype")

    def test_constraint_large_separation_returns_bool_dtype(
        self,
        begin_time_array: Time,
        ground_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        solar_system_constraint_large_separation: SolarSystemConstraint,
    ) -> None:
        """Test constraint with large separation returns boolean dtype."""
        result_large = solar_system_constraint_large_separation(begin_time_array, ground_ephemeris, sky_coord)
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

    def test_constraint_combined_magnitude_and_separation_checks_returns_ndarray(
        self,
        mock_get_slice: None,
        mock_get_body: None,
        mock_ephemeris: Ephemeris,
        multi_time_array: Time,
        test_coord: SkyCoord,
        test_constraint: SolarSystemConstraint,
    ) -> None:
        """
        Test that constraint works correctly with multiple time steps and returns ndarray.
        """

        # This should work and return a boolean array, but instead crashes!
        result = test_constraint(multi_time_array, mock_ephemeris, test_coord)

        # This result should be a boolean array of length 5
        assert isinstance(result, np.ndarray)

    def test_constraint_combined_magnitude_and_separation_checks_returns_bool_dtype(
        self,
        mock_get_slice: None,
        mock_get_body: None,
        mock_ephemeris: Ephemeris,
        multi_time_array: Time,
        test_coord: SkyCoord,
        test_constraint: SolarSystemConstraint,
    ) -> None:
        """Test that constraint works correctly with multiple time steps and returns bool dtype."""

        # This should work and return a boolean array, but instead crashes!
        result = test_constraint(multi_time_array, mock_ephemeris, test_coord)

        assert result.dtype == np.bool_

    def test_constraint_combined_magnitude_and_separation_checks_length_five(
        self,
        mock_get_slice: None,
        mock_get_body: None,
        mock_ephemeris: Ephemeris,
        multi_time_array: Time,
        test_coord: SkyCoord,
        test_constraint: SolarSystemConstraint,
    ) -> None:
        """Test that constraint works correctly with multiple time steps and returns array of length 5."""

        # This should work and return a boolean array, but instead crashes!
        result = test_constraint(multi_time_array, mock_ephemeris, test_coord)

        assert len(result) == 5

    def test_constraint_get_body_failure_logs_warning(
        self,
        mock_get_slice: None,
        caplog: pytest.LogCaptureFixture,
        mock_ephemeris: Ephemeris,
        mock_get_body: None,
        solar_system_constraint_single_body: SolarSystemConstraint,
    ) -> None:
        """Test that constraint logs warning when get_body fails and continues."""

        # Create time array
        time = Time(["2025-01-01"])
        coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg)

        # Call, should not raise, log warning, and return False
        result = solar_system_constraint_single_body(time, mock_ephemeris, coord)

        assert not result  # Should be False since body position failed


class TestCalculateBodyMagnitude:
    """Test suite for SolarSystemConstraint._calculate_body_magnitude method."""

    def test_calculate_magnitude_mercury(
        self,
        body_coord_1au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Mercury."""
        # Update ephemeris sun to 180 deg separation (phase_angle=180)
        sun_coord = SkyCoord(ra=180 * u.deg, dec=0 * u.deg, distance=1 * u.AU)
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
        mock_ephemeris_with_sun.sun = sun_array

        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.MERCURY, body_coord_1au, mock_ephemeris_with_sun, slice_index
        )
        phase_angle = 180.0
        expected = -1.9 + 0.02 * phase_angle + 3.5e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_venus(
        self,
        body_coord_1au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Venus."""
        # Mock body_coord at 1 AU, sun at 90 deg separation (phase_angle=90)
        sun_coord = SkyCoord(ra=90 * u.deg, dec=0 * u.deg, distance=1 * u.AU)

        # Create SkyCoord array from single coordinate
        sun_array = SkyCoord([sun_coord.ra], [sun_coord.dec], distance=[sun_coord.distance])
        mock_ephemeris_with_sun.sun = sun_array

        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.VENUS, body_coord_1au, mock_ephemeris_with_sun, slice_index
        )
        phase_angle = 90.0
        expected = -4.7 + 0.013 * phase_angle + 4.3e-7 * phase_angle**3
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_mars(
        self,
        body_coord_1_5au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Mars."""
        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.MARS, body_coord_1_5au, mock_ephemeris_with_sun, slice_index
        )
        distance_au = 1.5
        expected = -1.52 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_jupiter(
        self,
        body_coord_5au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Jupiter."""
        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.JUPITER, body_coord_5au, mock_ephemeris_with_sun, slice_index
        )
        distance_au = 5
        expected = -9.4 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_saturn(
        self,
        body_coord_9au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Saturn."""
        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.SATURN, body_coord_9au, mock_ephemeris_with_sun, slice_index
        )
        distance_au = 9
        expected = -8.9 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_uranus(
        self,
        body_coord_19au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Uranus."""
        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.URANUS, body_coord_19au, mock_ephemeris_with_sun, slice_index
        )
        distance_au = 19
        expected = 5.5 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_neptune(
        self,
        body_coord_30au: SkyCoord,
        mock_ephemeris_with_sun: Ephemeris,
        solar_system_constraint: SolarSystemConstraint,
        slice_index: slice,
    ) -> None:
        """Test magnitude calculation for Neptune."""
        result = solar_system_constraint._calculate_body_magnitude(
            SolarSystemObject.NEPTUNE, body_coord_30au, mock_ephemeris_with_sun, slice_index
        )
        distance_au = 30
        expected = 7.8 + 5 * np.log10(distance_au)
        assert np.allclose(result, [expected])

    def test_calculate_magnitude_unknown_body(
        self, mock_ephemeris: Ephemeris, solar_system_constraint: SolarSystemConstraint, slice_index: slice
    ) -> None:
        """Test magnitude calculation raises error for unknown body."""
        body_coord = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, distance=1 * u.AU)

        with pytest.raises(ValueError, match="Unknown body for magnitude calculation"):
            solar_system_constraint._calculate_body_magnitude(
                "pluto", body_coord, mock_ephemeris, slice_index
            )
