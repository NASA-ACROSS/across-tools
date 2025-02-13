import pytest
from astropy.coordinates import EarthLocation  # type: ignore[import-untyped]

from across.tools.ephemeris.ground_ephem import GroundEphemeris


class TestGroundEphemeris:
    """Test suite for the GroundEphemeris class."""

    def test_prepare_data_earth_location_success(self, ground_ephemeris: GroundEphemeris) -> None:
        """Test that prepare_data successfully computes the earth location."""
        assert isinstance(ground_ephemeris.earth_location, EarthLocation)

    def test_prepare_data_gcrs_success(self, ground_ephemeris: GroundEphemeris) -> None:
        """Test that prepare_data successfully computes the gcrs."""
        assert ground_ephemeris.gcrs is not None

    def test_prepare_data_no_location(self) -> None:
        """Test that prepare_data raises a ValueError if the location is not set."""
        begin = "2023-01-01T00:00:00"
        end = "2023-01-01T00:01:00"
        step_size = 60

        ephemeris = GroundEphemeris(begin, end, step_size)

        with pytest.raises(ValueError, match="Location of observatory not set"):
            ephemeris.prepare_data()
