import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.visibility.constraints.galactic_bulge import GalacticBulgeConstraint


@pytest.fixture
def galactic_bulge_constraint() -> GalacticBulgeConstraint:
    """Default galactic bulge constraint used in call tests."""
    return GalacticBulgeConstraint(min_separation=10.0)


@pytest.fixture
def bulge_center_coord() -> SkyCoord:
    """Coordinate at the galactic bulge center."""
    return SkyCoord(ra="17h45m40.04s", dec="-29d00m28.1s", frame="icrs")


@pytest.fixture
def near_bulge_coord(bulge_center_coord: SkyCoord) -> SkyCoord:
    """Coordinate close to the bulge center (within default threshold)."""
    return SkyCoord(ra=bulge_center_coord.ra + 1 * u.deg, dec=bulge_center_coord.dec)


@pytest.fixture
def far_from_bulge_coord() -> SkyCoord:
    """Coordinate far from the galactic bulge."""
    return SkyCoord(ra=0 * u.deg, dec=60 * u.deg)
