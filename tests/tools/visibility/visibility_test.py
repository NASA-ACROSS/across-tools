import uuid

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.core.schemas.visibility import VisibilityWindow
from across.tools.visibility import EphemerisVisibility, Visibility, compute_joint_visibility


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


class TestComputeJointVisibility:
    """Test the compute_joint_visibility function."""    
    def test_compute_joint_visibility_should_return_list(
        self, computed_joint_visibility: list[VisibilityWindow]
    ) -> None:
        """compute_joint_visibility should return a list."""
        assert isinstance(computed_joint_visibility, list)

    def test_compute_joint_visibility_window_should_be_not_empty(
        self, computed_joint_visibility: list[VisibilityWindow]
    ) -> None:
        """computed joint visibility should not be empty."""
        assert len(computed_joint_visibility) > 0

    def test_compute_joint_visibility_should_return_correct_type(
        self, computed_joint_visibility: list[VisibilityWindow]
    ) -> None:
        """compute_joint_visibility should return a list of EphemerisVisibilities."""
        assert isinstance(computed_joint_visibility[0], VisibilityWindow)

    @pytest.mark.parametrize(
        "field", 
        [
            "window",
            "max_visibility_duration",
            "constraint_reason",
        ]
    )
    def test_compute_joint_visibility_should_return_expected_result(
        self,
        computed_joint_visibility: list[VisibilityWindow],
        field: str,
        expected_joint_visibility_windows: list[VisibilityWindow],
    ) -> None:
        """Test that expected joint windows match calculated joint windows"""
        assert (
            computed_joint_visibility[0].model_dump()[field] 
            == 
            expected_joint_visibility_windows[0].model_dump()[field]
        )

    def test_compute_joint_visibility_should_return_empty_list_if_no_windows(
        self,
        computed_visibility: EphemerisVisibility,
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """
        compute joint visibility should return an empty list
        if any of the input windows are empty
        """
        computed_visibility_2 = computed_visibility
        computed_visibility_2.visibility_windows = []

        joint_visibility_windows = compute_joint_visibility(
            visibilities=[
                computed_visibility,
                computed_visibility_2
            ],
            instrument_ids=[test_observatory_id, test_observatory_id_2]
        )
        assert len(joint_visibility_windows) == 0

    def test_compute_joint_visibility_should_return_empty_list_if_no_overlap(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_no_overlap: EphemerisVisibility,
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """
        compute joint visibility should return an empty list
        if there is no overlap between the individual windows.
        """
        joint_visibility_windows = compute_joint_visibility(
            visibilities=[
                computed_visibility,
                computed_visibility_with_no_overlap,
            ],
            instrument_ids=[test_observatory_id, test_observatory_id_2]
        )
        assert len(joint_visibility_windows) == 0
