from datetime import datetime

import pytest

from across.tools.ephemeris.jpl_ephem import JPLEphemeris


class TestJPLEphemeris:
    """Test suite for the JPLEphemeris class."""

    def test_jpl_ephemeris_no_naif_id(self) -> None:
        """Test that JPLEphemeris raises ValueError when no NAIF ID is provided."""
        begin = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        with pytest.raises(ValueError, match="No NAIF ID provided"):
            ephemeris = JPLEphemeris(begin=begin, end=end)
            ephemeris.prepare_data()
