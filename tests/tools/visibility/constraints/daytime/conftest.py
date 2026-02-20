import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import Latitude, Longitude, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris.ground_ephem import GroundEphemeris


@pytest.fixture
def mauna_kea_latitude() -> Latitude:
    """Mauna Kea Observatory latitude."""
    return Latitude(19.8207 * u.deg)


@pytest.fixture
def mauna_kea_longitude() -> Longitude:
    """Mauna Kea Observatory longitude."""
    return Longitude(-155.4681 * u.deg)


@pytest.fixture
def mauna_kea_height() -> u.Quantity:
    """Mauna Kea Observatory height."""
    return 4205 * u.m


@pytest.fixture
def mauna_kea_ephemeris(
    mauna_kea_latitude: Latitude,
    mauna_kea_longitude: Longitude,
    mauna_kea_height: u.Quantity,
) -> GroundEphemeris:
    """Ground ephemeris for Mauna Kea Observatory over 24 hours."""
    start_time = Time("2024-06-15T00:00:00")
    end_time = Time("2024-06-16T00:00:00")
    step_size = 300

    ephem = GroundEphemeris(
        begin=start_time.datetime,
        end=end_time.datetime,
        step_size=step_size,
        latitude=mauna_kea_latitude,
        longitude=mauna_kea_longitude,
        height=mauna_kea_height,
    )
    ephem.compute()
    return ephem


@pytest.fixture
def dummy_coord() -> SkyCoord:
    """Create a dummy SkyCoord instance for testing."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg)
