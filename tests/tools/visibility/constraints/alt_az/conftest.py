from datetime import datetime

import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris


@pytest.fixture
def az_zero_alt_forty_five_sky_coord(ground_ephemeris: Ephemeris, ephemeris_begin: datetime) -> SkyCoord:
    """Fixture for a sky coordinate at 45 deg altitude and 50 deg azimuth."""
    return SkyCoord(
        AltAz(
            alt=45 * u.deg,
            az=50 * u.deg,
            location=ground_ephemeris.earth_location,
            obstime=ephemeris_begin,
        )
    )


@pytest.fixture
def az_eight_alt_five_sky_coord(ground_ephemeris: Ephemeris, ephemeris_begin: datetime) -> SkyCoord:
    """Fixture for a sky coordinate at 8 deg altitude and 5 deg azimuth."""
    return SkyCoord(
        AltAz(
            alt=8 * u.deg,
            az=5 * u.deg,
            location=ground_ephemeris.earth_location,
            obstime=ephemeris_begin,
        )
    )
