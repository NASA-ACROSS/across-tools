import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.visibility.constraints.galactic_plane import GalacticPlaneConstraint


@pytest.fixture
def galactic_plane_constraint() -> GalacticPlaneConstraint:
    """Default galactic plane constraint used in call tests."""
    return GalacticPlaneConstraint(min_latitude=10.0)


@pytest.fixture
def high_galactic_latitude_coord() -> SkyCoord:
    """Coordinate at high galactic latitude."""
    return SkyCoord(l=0 * u.deg, b=60 * u.deg, frame="galactic")


@pytest.fixture
def low_galactic_latitude_coord() -> SkyCoord:
    """Coordinate near the galactic plane."""
    return SkyCoord(l=0 * u.deg, b=5 * u.deg, frame="galactic")


@pytest.fixture
def galactic_equator_coord() -> SkyCoord:
    """Coordinate on the galactic equator."""
    return SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")


@pytest.fixture
def south_galactic_pole_coord() -> SkyCoord:
    """Coordinate at the south galactic pole."""
    return SkyCoord(l=0 * u.deg, b=-90 * u.deg, frame="galactic")
