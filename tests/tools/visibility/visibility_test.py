from datetime import datetime

import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from across.tools.visibility.base import Visibility


def test_validate_skycoord_type(mock_visibility: Visibility) -> None:
    """Test that SkyCoord is created from RA and Dec"""
    vis = mock_visibility
    assert isinstance(vis.skycoord, SkyCoord)


def test_validate_skycoord_ra(mock_visibility: Visibility, test_coords: tuple[float, float]) -> None:
    """Test that RA is correctly set"""
    vis = mock_visibility
    assert vis.ra == test_coords[0]


def test_validate_skycoord_dec(mock_visibility: Visibility, test_coords: tuple[float, float]) -> None:
    """Test that Dec is correctly set"""
    vis = mock_visibility
    assert vis.dec == test_coords[1]


def test_validate_skycoord_from_skycoord_dec(mock_visibility: Visibility) -> None:
    """Test that Dec is correctly set from SkyCoord"""
    vis = mock_visibility
    assert vis.dec == 45.0


def test_validate_step_size_type(mock_visibility: Visibility) -> None:
    """Test that step size is correctly set to TimeDelta"""
    vis = mock_visibility
    assert isinstance(vis.step_size, TimeDelta)


def test_compute_timestamp_not_none(mock_visibility: Visibility) -> None:
    """Test that timestamp is set after compute"""
    mock_visibility.compute()
    assert mock_visibility.timestamp is not None


def test_compute_entries_not_empty(mock_visibility: Visibility) -> None:
    """Test that entries are not empty after compute"""
    mock_visibility.compute()
    assert len(mock_visibility.entries) > 0


def test_visible_at_noon(mock_visibility: Visibility) -> None:
    """Test that target is visible at noon"""
    mock_visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))  # Noon on Jan 1
    assert mock_visibility.visible(t) is True


def test_index_type(mock_visibility: Visibility) -> None:
    """Test that index returns an integer"""
    mock_visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))
    idx = mock_visibility.index(t)
    assert isinstance(idx, int)


def test_index_non_negative(mock_visibility: Visibility) -> None:
    """Test that index is non-negative"""
    mock_visibility.compute()
    t = Time(datetime(2023, 1, 1, 12))
    idx = mock_visibility.index(t)
    assert idx >= 0


def test_index_error_without_compute(mock_visibility: Visibility) -> None:
    """Test that index raises error without compute"""
    with pytest.raises(ValueError):
        mock_visibility.index(Time(datetime(2023, 1, 1, 12)))
