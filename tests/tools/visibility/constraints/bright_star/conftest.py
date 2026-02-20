from collections.abc import Generator
from unittest.mock import patch

import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.visibility.constraints.bright_star import BrightStarConstraint


@pytest.fixture
def mock_get_bright_stars(
    mock_bright_stars: list[tuple[SkyCoord, float]],
) -> Generator[list[tuple[SkyCoord, float]], None, None]:
    """Patch get_bright_stars to prevent internet access during tests."""
    with patch("across.tools.visibility.constraints.bright_star.get_bright_stars") as mock:
        mock.return_value = mock_bright_stars
        yield mock


@pytest.fixture
def bright_star_constraint() -> BrightStarConstraint:
    """Default bright star constraint used in call tests."""
    return BrightStarConstraint(min_separation=5.0)


@pytest.fixture
def sirius_coord() -> SkyCoord:
    """Coordinate at Sirius position."""
    return SkyCoord(ra="06h45m08.9s", dec="-16d42m58.0s")


@pytest.fixture
def near_sirius_coord(sirius_coord: SkyCoord) -> SkyCoord:
    """Coordinate near Sirius (within default threshold)."""
    return SkyCoord(ra=sirius_coord.ra + 1 * u.deg, dec=sirius_coord.dec)


@pytest.fixture
def far_from_bright_stars_coord() -> SkyCoord:
    """Coordinate far from bright stars."""
    return SkyCoord(ra=0 * u.deg, dec=80 * u.deg)


@pytest.fixture
def mixed_bright_star_coords() -> SkyCoord:
    """Coordinates containing one point at Sirius."""
    return SkyCoord(
        ra=["00h00m00s", "06h45m08.9s", "10h00m00s"],
        dec=["00d00m00s", "-16d42m58.0s", "00d00m00s"],
    )


@pytest.fixture
def mixed_bright_star_times() -> Time:
    """Timestamps aligned with mixed bright star coordinates."""
    return Time(["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"])
