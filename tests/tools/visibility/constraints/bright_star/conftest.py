from collections.abc import Callable, Generator
from unittest.mock import MagicMock

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.visibility.constraints.bright_star import BrightStarConstraint


@pytest.fixture
def mock_get_bright_stars(
    mock_bright_stars: list[tuple[SkyCoord, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[MagicMock, None, None]:
    """Patch get_bright_star_coordinates to prevent internet access during tests."""
    mock = MagicMock(return_value=SkyCoord([star for star, _ in mock_bright_stars]))
    monkeypatch.setattr("across.tools.visibility.constraints.bright_star.get_bright_star_coordinates", mock)
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


@pytest.fixture
def sky_grid_coords(sirius_coord: SkyCoord) -> SkyCoord:
    """A grid of coordinates across the sky, plus points at increasing distances from Sirius."""
    ra, dec = np.meshgrid(np.arange(0, 360, 30), np.arange(-75, 90, 30))
    grid = SkyCoord(ra=ra.ravel() * u.deg, dec=dec.ravel() * u.deg)
    near_sirius = sirius_coord.directional_offset_by(0 * u.deg, [0.5, 2, 4.9, 5.1, 10, 25] * u.deg)
    return SkyCoord([*grid, *near_sirius])


@pytest.fixture
def sky_grid_times(sky_grid_coords: SkyCoord) -> Time:
    """Timestamps aligned with the sky grid coordinates."""
    return Time("2024-01-01T00:00:00") + np.arange(len(sky_grid_coords)) * u.min


@pytest.fixture
def per_star_result_factory(
    mock_bright_stars: list[tuple[SkyCoord, float]],
) -> Callable[[SkyCoord, float], np.typing.NDArray[np.bool_]]:
    """Check whether each coordinate is within min_separation of any bright star, one star at a time."""

    def _build(coordinates: SkyCoord, min_separation: float) -> np.typing.NDArray[np.bool_]:
        return np.array(
            [
                any(coordinate.separation(star) < min_separation * u.deg for star, _ in mock_bright_stars)
                for coordinate in coordinates
            ]
        )

    return _build
