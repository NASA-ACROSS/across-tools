from datetime import datetime

import pytest

from across.tools.ephemeris.tle_ephem import TLEEphemeris


class TestTLEEphemeris:
    """Test suite for the TLEEphemeris class."""

    def test_tle_ephemeris_no_naif_id(self) -> None:
        """Test that TLEEphemeris raises ValueError when no NAIF ID is provided."""
        begin = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        with pytest.raises(ValueError, match="No TLE provided"):
            ephemeris = TLEEphemeris(begin=begin, end=end)
            ephemeris.prepare_data()
