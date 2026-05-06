from collections.abc import Generator
from datetime import datetime
from typing import Any, Callable
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
def multi_norad_spacetrack_tle_response() -> str:
    """Return a TLE response containing multiple NORAD IDs and repeated entries."""
    return (
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        "1 28485U 05055A   08263.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 28485  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        "1 25544U 98067A   08262.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )


@pytest.fixture
def duplicate_norad_spacetrack_tle_response() -> str:
    """Return a TLE response with multiple entries for the same NORAD ID at different epochs."""
    return (
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        "1 25544U 98067A   08263.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        "1 25544U 98067A   08262.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )


@pytest.fixture
def empty_spacetrack_tle_response() -> str:
    """Return an empty TLE response."""
    return ""


@pytest.fixture
def configure_mock_spacetrack_gp(mock_spacetrack: MagicMock) -> Callable[[str], MagicMock]:
    """Configure the mocked SpaceTrack client gp() response and return the client."""

    def _configure(response: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.gp.return_value = response
        mock_spacetrack.return_value.__enter__.return_value = mock_client
        return mock_client

    return _configure


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
        satellites=[{"name": "ISS", "id": 25544}],
        epoch=datetime(2008, 9, 20),
        spacetrack_user="user",
        spacetrack_pwd="pass",
    )
