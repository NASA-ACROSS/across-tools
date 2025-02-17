from datetime import datetime

import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.visibility.base import Visibility


class TestVisibility:
    """Test the Visibility class"""

    def test_validate_skycoord_type(self, mock_visibility: Visibility) -> None:
        """Test that SkyCoord is created from RA and Dec"""
        vis = mock_visibility
        assert isinstance(vis.skycoord, SkyCoord)

    def test_validate_skycoord_ra(
        self, mock_visibility: Visibility, test_coords: tuple[float, float]
    ) -> None:
        """Test that RA is correctly set"""
        vis = mock_visibility
        assert vis.ra == test_coords[0]

    def test_validate_skycoord_dec(
        self, mock_visibility: Visibility, test_coords: tuple[float, float]
    ) -> None:
        """Test that Dec is correctly set"""
        vis = mock_visibility
        assert vis.dec == test_coords[1]

    def test_validate_skycoord_from_skycoord_dec(self, mock_visibility: Visibility) -> None:
        """Test that Dec is correctly set from SkyCoord"""
        vis = mock_visibility
        assert vis.dec == 45.0

    def test_validate_step_size_type(self, mock_visibility: Visibility) -> None:
        """Test that step size is correctly set to TimeDelta"""
        vis = mock_visibility
        assert isinstance(vis.step_size, TimeDelta)

    def test_compute_timestamp_not_none(self, mock_visibility: Visibility) -> None:
        """Test that timestamp is set after compute"""
        mock_visibility.compute()
        assert mock_visibility.timestamp is not None

    def test_compute_entries_not_empty(self, mock_visibility: Visibility) -> None:
        """Test that entries are not empty after compute"""
        mock_visibility.compute()
        assert len(mock_visibility.entries) > 0

    def test_visible_at_noon(self, mock_visibility: Visibility) -> None:
        """Test that target is visible at noon"""
        mock_visibility.compute()
        t = Time(datetime(2023, 1, 1, 12))  # Noon on Jan 1
        assert mock_visibility.visible(t) is True

    def test_index_type(self, mock_visibility: Visibility) -> None:
        """Test that index returns an integer"""
        mock_visibility.compute()
        t = Time(datetime(2023, 1, 1, 12))
        idx = mock_visibility.index(t)
        assert isinstance(idx, int)

    def test_index_non_negative(self, mock_visibility: Visibility) -> None:
        """Test that index is non-negative"""
        mock_visibility.compute()
        t = Time(datetime(2023, 1, 1, 12))
        idx = mock_visibility.index(t)
        assert idx >= 0

    def test_index_error_without_compute(self, mock_visibility: Visibility) -> None:
        """Test that index raises error without compute"""
        with pytest.raises(ValueError):
            mock_visibility.index(Time(datetime(2023, 1, 1, 12)))

    def test_timestamp_not_set_exception_in_make_windows(self, mock_visibility: Visibility) -> None:
        """Test that timestamp is set before make_windows"""
        with pytest.raises(ValueError):
            mock_visibility._make_windows()

    def test_step_size_cannot_be_negative(self, test_skycoord: SkyCoord) -> None:
        """Test that step size cannot be negative"""
        with pytest.raises(ValidationError) as excinfo:

            class Dog(Visibility):
                pass

                def _constraint(self, i: int) -> str:
                    return "test"

                def prepare_data(self) -> None:
                    pass

            begin, end = Time(datetime(2023, 1, 1)), Time(datetime(2023, 1, 2))

            Dog(begin=begin, end=end, skycoord=test_skycoord, step_size=TimeDelta(-1 * u.s))
        assert "must be a positive" in str(excinfo.value)

    def test_step_size_int(self, mock_visibility_step_size_int: Visibility) -> None:
        """Test that step size argument can be an integer"""
        assert isinstance(mock_visibility_step_size_int.step_size, TimeDelta)

    def test_step_size_datetime_timedelta(
        self, mock_visibility_step_size_datetime_timedelta: Visibility
    ) -> None:
        """Test that step size argument can be a datetime timedelta"""
        assert isinstance(mock_visibility_step_size_datetime_timedelta.step_size, TimeDelta)
