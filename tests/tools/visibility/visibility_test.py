import uuid

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.visibility import Visibility


class TestVisibility:
    """Test the Visibility class"""

    def test_validate_skycoord_type(self, mock_visibility: Visibility) -> None:
        """Test that SkyCoord is created from RA and Dec"""
        vis = mock_visibility
        assert isinstance(vis.coordinate, SkyCoord)

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
        assert len(mock_visibility.visibility_windows) > 0

    def test_visible_at_noon(self, mock_visibility: Visibility, noon_time: Time) -> None:
        """Test that target is visible at noon"""
        mock_visibility.compute()
        assert mock_visibility.visible(noon_time) is True

    def test_not_visible_at_midnight(self, mock_visibility: Visibility, midnight_time: Time) -> None:
        """Test that target is not visible at midnight"""
        mock_visibility.compute()
        assert mock_visibility.visible(midnight_time) is False

    def test_visible_at_noon_and_a_few_minutes_later(
        self, mock_visibility: Visibility, noon_time_array: Time
    ) -> None:
        """Test that verifies functionality of Ephemeris visible method with an
        array of times that are all visible"""
        mock_visibility.compute()
        assert np.all(mock_visibility.visible(noon_time_array)) is np.True_

    def test_visible_over_earth_limb(self, computed_visibility: Visibility) -> None:
        """Test that verifies functionality of Ephemeris visible method with an
        array of times that include times when the target is not visible"""
        assert isinstance(computed_visibility.timestamp, Time)
        times = computed_visibility.timestamp[0:10]
        assert np.all(computed_visibility.visible(times)) is np.False_

    def test_index_type(self, mock_visibility: Visibility, noon_time: Time) -> None:
        """Test that index returns an integer"""
        mock_visibility.compute()
        idx = mock_visibility.index(noon_time)
        assert isinstance(idx, int)

    def test_index_error_without_compute(self, mock_visibility: Visibility, noon_time: Time) -> None:
        """Test that index raises error without compute"""
        with pytest.raises(ValueError):
            mock_visibility.index(noon_time)

    def test_timestamp_not_set_exception_in_make_windows(self, mock_visibility: Visibility) -> None:
        """Test that timestamp is set before make_windows"""
        with pytest.raises(ValueError):
            mock_visibility._make_windows()

    def test_step_size_cannot_be_negative(
        self,
        test_skycoord: SkyCoord,
        mock_visibility_class: type[Visibility],
        test_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_observatory_id: uuid.UUID,
        test_observatory_name: str,
    ) -> None:
        """Test that step size cannot be negative"""
        with pytest.raises(ValidationError) as excinfo:
            begin, end = test_time_range

            mock_visibility_class(
                begin=begin,
                end=end,
                coordinate=test_skycoord,
                step_size=-test_step_size,
                observatory_id=test_observatory_id,
                observatory_name=test_observatory_name,
            )
        assert "must be a positive" in str(excinfo.value)

    def test_step_size_int(self, mock_visibility_step_size_int: Visibility) -> None:
        """Test that step size argument can be an integer"""
        assert isinstance(mock_visibility_step_size_int.step_size, TimeDelta)

    def test_step_size_datetime_timedelta(
        self, mock_visibility_step_size_datetime_timedelta: Visibility
    ) -> None:
        """Test that step size argument can be a datetime timedelta"""
        assert isinstance(mock_visibility_step_size_datetime_timedelta.step_size, TimeDelta)

    def test_no_coordinate_given(
        self,
        test_time_range: tuple[Time, Time],
        mock_visibility_class: type[Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_name: str,
    ) -> None:
        """Test that step size cannot be negative"""
        with pytest.raises(ValidationError) as excinfo:
            begin, end = test_time_range
            mock_visibility_class(
                begin=begin,
                end=end,
                step_size=1 * u.s,
                observatory_id=test_observatory_id,
                observatory_name=test_observatory_name,
            )
        assert "Must supply either coordinate" in str(excinfo.value)

    def test_no_step_size_given_gives_default(
        self,
        test_time_range: tuple[Time, Time],
        test_skycoord: SkyCoord,
        mock_visibility_class: type[Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_name: str,
        default_step_size: TimeDelta,
    ) -> None:
        """Test that step size is set to default if not given"""

        dog = mock_visibility_class(
            begin=test_time_range[0],
            end=test_time_range[1],
            coordinate=test_skycoord,
            observatory_id=test_observatory_id,
            observatory_name=test_observatory_name,
        )
        assert dog.step_size == default_step_size
