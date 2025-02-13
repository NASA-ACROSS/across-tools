from datetime import datetime

import pytest

from across.tools.ephemeris.spice_ephem import SPICEEphemeris


class TestSPICEEphemeris:
    """Test suite for the SPICEEphemeris class."""

    def test_spice_ephemeris_no_kernel_url(self) -> None:
        """Test that prepare_data raises a ValueError if the kernel URL is not set."""
        begin = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        with pytest.raises(ValueError, match="No SPICE kernel URL provided"):
            SPICEEphemeris(begin=begin, end=end, step_size=60, naif_id=301).prepare_data()

    def test_spice_ephemeris_no_naif_id(self) -> None:
        """Test that prepare_data raises a ValueError if the NAIF ID is not set."""
        begin = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)
        with pytest.raises(ValueError, match="No NAIF ID provided"):
            SPICEEphemeris(
                begin=begin, end=end, step_size=60, spice_kernel_url="https://example.com/fake.bsp"
            ).prepare_data()
