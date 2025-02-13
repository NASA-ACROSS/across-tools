from datetime import timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from numpy.typing import NDArray

from across.tools.ephemeris import Ephemeris


class TestEphemeris:
    """Test the Ephemeris class."""

    def test_abc_class_not_implementation(
        self,
        not_implemented_ephemeris_class: type[Ephemeris],
        ephemeris_begin: Time,
        ephemeris_end: Time,
        ephemeris_step_size: TimeDelta,
    ) -> None:
        """Test that the Ephemeris class is an abstract base class."""

        test = not_implemented_ephemeris_class(
            begin=ephemeris_begin, end=ephemeris_end, step_size=ephemeris_step_size
        )
        with pytest.raises(NotImplementedError):
            test.prepare_data()

    def test_compute_ground_ephemeris_instance(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that compute_ground_ephemeris returns an Ephemeris object."""
        assert isinstance(keck_ground_ephemeris, Ephemeris)

    def test_compute_ground_ephemeris_timestamp_length(self, keck_ground_ephemeris: Ephemeris) -> None:
        """Test that the timestamp has a length of 6."""
        assert len(keck_ground_ephemeris.timestamp) == 6

    def test_compute_ground_ephemeris_latitude(
        self, keck_ground_ephemeris: Ephemeris, keck_latitude: u.Quantity
    ) -> None:
        """Test the latitude of the ground ephemeris."""
        assert np.isclose(
            keck_ground_ephemeris.earth_location.lat.to_value(u.deg), keck_latitude.to_value(u.deg), 1e-3
        )

    def test_compute_ground_ephemeris_longitude(
        self, keck_ground_ephemeris: Ephemeris, keck_longitude: u.Quantity
    ) -> None:
        """Test the longitude of the ground ephemeris."""
        assert np.isclose(
            keck_ground_ephemeris.earth_location.lon.to_value(u.deg), keck_longitude.to_value(u.deg), 1e-3
        )

    def test_compute_ground_ephemeris_height(
        self, keck_ground_ephemeris: Ephemeris, keck_height: u.Quantity
    ) -> None:
        """Test the height of the ground ephemeris."""
        assert np.isclose(
            keck_ground_ephemeris.earth_location.height.to_value(u.m), keck_height.to_value(u.m), 1e-3
        )

    def test_compute_tle_ephemeris_instance(self, hubble_tle_ephemeris: Ephemeris) -> None:
        """Test that compute_tle_ephemeris returns an Ephemeris object."""
        assert isinstance(hubble_tle_ephemeris, Ephemeris)

    def test_compute_tle_ephemeris_timestamp_length(self, hubble_tle_ephemeris: Ephemeris) -> None:
        """Test that the timestamp has a length of 6."""
        assert len(hubble_tle_ephemeris.timestamp) == 6

    def test_compute_tle_ephemeris_gcrs(
        self, hubble_tle_ephemeris: Ephemeris, hubble_gcrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the GCRS coordinates from the TLE ephemeris."""
        assert np.allclose(hubble_tle_ephemeris.gcrs.cartesian.xyz.to_value(u.km), hubble_gcrs_value_km, 1e-3)

    def test_compute_tle_ephemeris_itrs(
        self, hubble_tle_ephemeris: Ephemeris, hubble_itrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the ITRS coordinates from the TLE ephemeris."""
        assert np.allclose(
            hubble_tle_ephemeris.earth_location.itrs.cartesian.xyz.to_value(u.km), hubble_itrs_value_km, 1e-3
        )

    def test_compute_jpl_ephemeris_instance(self, hubble_jpl_ephemeris: Ephemeris) -> None:
        """Test that compute_jpl_ephemeris returns an Ephemeris object."""
        assert isinstance(hubble_jpl_ephemeris, Ephemeris)

    def test_compute_jpl_ephemeris_timestamp_length(self, hubble_jpl_ephemeris: Ephemeris) -> None:
        """Test that the timestamp has a length of 6."""
        assert len(hubble_jpl_ephemeris.timestamp) == 6

    def test_compute_jpl_ephemeris_gcrs(
        self, hubble_jpl_ephemeris: Ephemeris, hubble_gcrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the GCRS coordinates from the JPL ephemeris."""
        assert np.allclose(hubble_jpl_ephemeris.gcrs.cartesian.xyz.to_value(u.km), hubble_gcrs_value_km, 1e-3)

    def test_compute_jpl_ephemeris_itrs(
        self, hubble_jpl_ephemeris: Ephemeris, hubble_itrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the ITRS coordinates from the JPL ephemeris."""
        assert np.allclose(
            hubble_jpl_ephemeris.earth_location.itrs.cartesian.xyz.to_value(u.km), hubble_itrs_value_km, 1
        )

    def test_compute_spice_ephemeris_instance(self, hubble_spice_ephemeris: Ephemeris) -> None:
        """Test that compute_spice_ephemeris returns an Ephemeris object."""
        assert isinstance(hubble_spice_ephemeris, Ephemeris)

    def test_compute_spice_ephemeris_timestamp_length(self, hubble_spice_ephemeris: Ephemeris) -> None:
        """Test that the timestamp has a length of 6."""
        assert len(hubble_spice_ephemeris.timestamp) == 6

    def test_compute_spice_ephemeris_gcrs(
        self, hubble_spice_ephemeris: Ephemeris, hubble_gcrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the GCRS coordinates from the SPICE ephemeris."""
        assert np.allclose(
            hubble_spice_ephemeris.gcrs.cartesian.xyz.to_value(u.km), hubble_gcrs_value_km, 1e-3
        )

    def test_compute_spice_ephemeris_itrs(
        self, hubble_spice_ephemeris: Ephemeris, hubble_itrs_value_km: NDArray[np.float64]
    ) -> None:
        """Test the ITRS coordinates from the SPICE ephemeris."""
        assert np.allclose(
            hubble_spice_ephemeris.earth_location.itrs.cartesian.xyz.to_value(u.km), hubble_itrs_value_km, 1
        )

    def test_ephemeris_init_time_objects_begin(
        self,
        mock_ephemeris: Ephemeris,
        ephemeris_begin: Time,
    ) -> None:
        """Test the initialization of the Ephemeris class with Time objects - begin."""
        assert mock_ephemeris.begin == ephemeris_begin

    def test_ephemeris_init_time_objects_end(
        self,
        mock_ephemeris: Ephemeris,
        ephemeris_end: Time,
    ) -> None:
        """Test the initialization of the Ephemeris class with Time objects - end."""
        assert mock_ephemeris.end == ephemeris_end

    def test_ephemeris_init_time_objects_step_size(
        self,
        mock_ephemeris: Ephemeris,
        ephemeris_step_size: TimeDelta,
    ) -> None:
        """Test the initialization of the Ephemeris class with Time objects - step size."""
        assert mock_ephemeris.step_size == ephemeris_step_size

    def test_ephemeris_init_datetime_objects_begin(self, mock_ephemeris: Ephemeris) -> None:
        """Test the initialization of the Ephemeris class with datetime objects - begin."""
        assert isinstance(mock_ephemeris.begin, Time)

    def test_ephemeris_init_datetime_objects_end(self, mock_ephemeris: Ephemeris) -> None:
        """Test the initialization of the Ephemeris class with datetime objects - end."""
        assert isinstance(mock_ephemeris.end, Time)

    def test_ephemeris_init_datetime_objects_step_size(self, mock_ephemeris: Ephemeris) -> None:
        """Test the initialization of the Ephemeris class with datetime objects - step size."""
        assert mock_ephemeris.step_size == TimeDelta(60 * u.s)

    def test_ephemeris_init_step_size_int(self, mock_ephemeris: Ephemeris) -> None:
        """Test the initialization of the Ephemeris class with int step size."""
        assert mock_ephemeris.step_size == TimeDelta(60 * u.s)

    def test_ephemeris_len(self, mock_ephemeris: Ephemeris) -> None:
        """Test the length of the Ephemeris class."""
        assert len(mock_ephemeris) == 6

    def test_ephemeris_len_single_date(
        self, mock_ephemeris_class: type[Ephemeris], ephemeris_begin: Time, ephemeris_step_size: TimeDelta
    ) -> None:
        """Test the length of the Ephemeris class when begin and end are the same."""
        ephem = mock_ephemeris_class(ephemeris_begin, ephemeris_begin, ephemeris_step_size)
        assert len(ephem) == 1

    def test_ephemeris_index(
        self, ephemeris_begin: Time, mock_ephemeris: Ephemeris, ephemeris_step_size: TimeDelta
    ) -> None:
        """Test the index method of the Ephemeris class."""
        t = ephemeris_begin + ephemeris_step_size * 3
        assert mock_ephemeris.index(t) == 3

    def test_ephemeris_index_outside_range(self, mock_ephemeris: Ephemeris) -> None:
        """Test the index method of the Ephemeris class with a time outside the
        range."""
        t = Time("2023-01-01T00:05:00")
        with pytest.raises(AssertionError):
            mock_ephemeris.index(t)

    def test_compute_timestamp_length(
        self, ephemeris_begin: Time, ephemeris_end: Time, mock_ephemeris: Ephemeris
    ) -> None:
        """Test the length of timestamps from _compute_timestamp method."""
        timestamps = mock_ephemeris._compute_timestamp()
        assert len(timestamps) == 6

    def test_compute_timestamp_first_value(
        self, ephemeris_begin: Time, ephemeris_end: Time, mock_ephemeris: Ephemeris
    ) -> None:
        """Test the first timestamp value from _compute_timestamp method."""
        timestamps = mock_ephemeris._compute_timestamp()
        assert timestamps[0] == ephemeris_begin

    def test_compute_timestamp_last_value(
        self, ephemeris_begin: Time, ephemeris_end: Time, mock_ephemeris: Ephemeris
    ) -> None:
        """Test the last timestamp value from _compute_timestamp method."""
        timestamps = mock_ephemeris._compute_timestamp()
        assert timestamps[-1] == ephemeris_end

    def test_compute_timestamp_single_length(
        self, mock_ephemeris_class: type[Ephemeris], ephemeris_begin: Time, ephemeris_step_size: TimeDelta
    ) -> None:
        """Test the length of timestamps when begin equals end."""
        ephem = mock_ephemeris_class(ephemeris_begin, ephemeris_begin, ephemeris_step_size)
        timestamps = ephem._compute_timestamp()
        assert len(timestamps) == 1

    def test_compute_timestamp_single_value(
        self, mock_ephemeris_class: type[Ephemeris], ephemeris_begin: Time, ephemeris_step_size: TimeDelta
    ) -> None:
        """Test the timestamp value when begin equals end."""
        ephem = mock_ephemeris_class(ephemeris_begin, ephemeris_begin, ephemeris_step_size)
        timestamps = ephem._compute_timestamp()
        assert timestamps[0] == ephemeris_begin

    def test_prepare_data(self, mock_ephemeris: Ephemeris) -> None:
        """Test the prepare_data method."""

        mock_ephemeris.prepare_data()  # Call prepare_data to ensure it runs without errors
        assert True  # If it runs without errors, the test passes

    def test_step_size_parameter_is_int(
        self, mock_ephemeris_class: type[Ephemeris], ephemeris_begin: Time, ephemeris_end: Time
    ) -> None:
        """Test the step_size attribute with an int."""
        ephem = mock_ephemeris_class(begin=ephemeris_begin, end=ephemeris_end, step_size=60)
        assert ephem.step_size == TimeDelta(60 * u.s)

    def test_step_size_parameter_is_datetime(
        self, mock_ephemeris_class: type[Ephemeris], ephemeris_begin: Time, ephemeris_end: Time
    ) -> None:
        """Test the step_size attribute with a datetime."""
        ephem = mock_ephemeris_class(
            begin=ephemeris_begin, end=ephemeris_end, step_size=timedelta(seconds=60)
        )
        assert ephem.step_size == TimeDelta(60 * u.s)

    @pytest.mark.parametrize("i", range(6))
    def test_every_index_near_values(self, mock_ephemeris: Ephemeris, i: int) -> None:
        """Test the index method for every timestamp with a small random offset."""
        t = mock_ephemeris.timestamp[i] + TimeDelta((0.2 ** np.random.random() - 0.1) * u.s)
        assert mock_ephemeris.index(t) == i
