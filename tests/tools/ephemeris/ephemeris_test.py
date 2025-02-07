import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from across.tools.ephemeris import Ephemeris


class TestEphemeris:
    """Test the Ephemeris class."""

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
