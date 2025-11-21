import uuid
from datetime import datetime

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.schemas import VisibilityWindow
from across.tools.visibility import (
    EphemerisVisibility,
    JointVisibility,
    compute_joint_visibility,
)


class TestJointVisibility:
    """Test the JointVisibility class"""

    def test_joint_visibility_should_raise_error_if_ras_differ(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Should raise ValueError if RA coordinates for input Visibilities are not the same"""
        computed_visibility_with_overlap.ra = 123.456
        with pytest.raises(ValueError):
            JointVisibility(
                visibilities=[computed_visibility, computed_visibility_with_overlap],
                instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            )

    def test_joint_visibility_should_raise_error_if_decs_differ(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Should raise ValueError if dec coordinates for input Visibilities are not the same"""
        computed_visibility_with_overlap.dec = -54.321
        with pytest.raises(ValueError):
            JointVisibility(
                visibilities=[computed_visibility, computed_visibility_with_overlap],
                instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            )

    def test_joint_visibility_should_raise_error_if_begin_times_differ(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Should raise ValueError if begin times for input Visibilities are not the same"""
        computed_visibility_with_overlap.begin = datetime(2025, 11, 15, 1, 1, 1)
        with pytest.raises(ValueError):
            JointVisibility(
                visibilities=[computed_visibility, computed_visibility_with_overlap],
                instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            )

    def test_joint_visibility_should_raise_error_if_end_times_differ(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Should raise ValueError if end times for input Visibilities are not the same"""
        computed_visibility_with_overlap.end = datetime(2025, 11, 15, 1, 11, 1)
        with pytest.raises(ValueError):
            JointVisibility(
                visibilities=[computed_visibility, computed_visibility_with_overlap],
                instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            )

    def test_joint_visibility_should_raise_error_if_step_sizes_differ(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Should raise ValueError if step sizes for input Visibilities are not the same"""
        computed_visibility_with_overlap.step_size = 120
        with pytest.raises(ValueError):
            JointVisibility(
                visibilities=[computed_visibility, computed_visibility_with_overlap],
                instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            )

    def test_compute_timestamp_not_none(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Test that timestamp is set after compute"""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        joint_vis.compute()
        assert joint_vis.timestamp is not None

    def test_compute_entries_not_empty(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Test that entries are not empty after compute"""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        joint_vis.compute()
        assert len(joint_vis.visibility_windows) > 0

    def test_visible_at_noon(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
        noon_time: Time,
    ) -> None:
        """Test that target is not visible at noon"""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        joint_vis.compute()
        assert joint_vis.visible(noon_time) is False

    def test_not_visible_at_midnight(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
        midnight_time: Time,
    ) -> None:
        """Test that target is visible at midnight"""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
        )
        joint_vis.compute()
        assert joint_vis.visible(midnight_time) is True


class TestComputeJointVisibility:
    """Test the compute_joint_visibility function."""

    def test_compute_joint_visibility_should_return_joint_visibility(
        self,
        computed_joint_visibility: JointVisibility[EphemerisVisibility],
    ) -> None:
        """compute_joint_visibility should return a JointVisibility object."""
        assert isinstance(computed_joint_visibility, JointVisibility)

    def test_compute_joint_visibility_window_should_be_not_empty(
        self,
        computed_joint_visibility: JointVisibility[EphemerisVisibility],
    ) -> None:
        """computed joint visibility windows should not be empty."""
        assert len(computed_joint_visibility.visibility_windows) > 0

    def test_compute_joint_visibility_should_return_correct_type(
        self,
        computed_joint_visibility: JointVisibility[EphemerisVisibility],
    ) -> None:
        """compute_joint_visibility should contain a list of EphemerisVisibilities."""
        assert isinstance(computed_joint_visibility.visibility_windows[0], VisibilityWindow)

    @pytest.mark.parametrize(
        "field",
        [
            "window",
            "max_visibility_duration",
            "constraint_reason",
        ],
    )
    def test_compute_joint_visibility_should_return_expected_result(
        self,
        computed_joint_visibility: JointVisibility[EphemerisVisibility],
        field: str,
        expected_joint_visibility_windows: list[VisibilityWindow],
    ) -> None:
        """Expected joint windows should match calculated joint windows"""
        assert (
            computed_joint_visibility.visibility_windows[0].model_dump()[field]
            == expected_joint_visibility_windows[0].model_dump()[field]
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
        computed_visibility_2.inconstraint = np.asarray(
            [True for i in range(len(computed_visibility_2.inconstraint))]
        )

        joint_visibility_windows = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_2],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        assert len(joint_visibility_windows.visibility_windows) == 0

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
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        assert len(joint_visibility_windows.visibility_windows) == 0
