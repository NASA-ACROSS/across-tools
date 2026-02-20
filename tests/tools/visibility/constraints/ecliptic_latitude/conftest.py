import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.visibility.constraints.ecliptic_latitude import EclipticLatitudeConstraint


@pytest.fixture
def ecliptic_constraint() -> EclipticLatitudeConstraint:
    """Default ecliptic latitude constraint used by call tests."""
    return EclipticLatitudeConstraint(min_latitude=15.0)


@pytest.fixture
def high_latitude_coord() -> SkyCoord:
    """Coordinate far from the ecliptic plane."""
    return SkyCoord(ra=90 * u.deg, dec=60 * u.deg)


@pytest.fixture
def ecliptic_equator_coord() -> SkyCoord:
    """Coordinate on the ecliptic plane (approximately 0 deg latitude)."""
    return SkyCoord(ra=0 * u.deg, dec=0 * u.deg)


@pytest.fixture
def ten_degree_latitude_coord() -> SkyCoord:
    """Coordinate around 10 deg ecliptic latitude for threshold tests."""
    return SkyCoord(ra=90 * u.deg, dec=10 * u.deg)


@pytest.fixture
def ecliptic_pole_coord() -> SkyCoord:
    """Coordinate at the ecliptic pole."""
    return SkyCoord(ra=90 * u.deg, dec=90 * u.deg)
