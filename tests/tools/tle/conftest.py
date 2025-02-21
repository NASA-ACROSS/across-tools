from typing import Any

import pytest


@pytest.fixture
def valid_tle_data() -> dict[str, Any]:
    """Fixture providing valid TLE data."""
    return {
        "norad_id": 25544,
        "satellite_name": "ISS (ZARYA)",
        "tle1": "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "tle2": "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    }
