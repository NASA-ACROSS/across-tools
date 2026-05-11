from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from across.tools.tle.tle import TLEFetch


@pytest.fixture
def mock_spacetrack() -> Generator[MagicMock]:
    """Return a mock SpaceTrackClient."""
    with patch("across.tools.tle.tle.SpaceTrackClient") as mock:
        yield mock


@pytest.fixture
def valid_spacetrack_tle_response() -> str:
    """Return a valid TLE response."""
    return (
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )


@pytest.fixture
def valid_tle_data(valid_spacetrack_tle_response: str) -> dict[str, Any]:
    """Fixture providing valid TLE data."""
    return {
        "norad_id": 25544,
        "satellite_name": "ISS (ZARYA)",
        "tle1": valid_spacetrack_tle_response.split("\n")[0],
        "tle2": valid_spacetrack_tle_response.split("\n")[1],
    }


@pytest.fixture
def tle_fetch_object() -> Generator[TLEFetch]:
    """Example TLEFetch object."""
    yield TLEFetch(
        norad_id=25544,
        epoch=datetime(2008, 9, 20),
        satellite_name="ISS",
        spacetrack_user="user",
        spacetrack_pwd="pass",
    )
