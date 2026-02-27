import uuid
from datetime import datetime

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas import VisibilityWindow
from across.tools.visibility import (
    EphemerisVisibility,
    JointVisibility,
    Visibility,
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

    def test_min_vis_excludes_short_windows(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """JointVisibility with min_vis larger than all window durations should yield no windows."""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            min_vis=999999,
        )
        joint_vis.compute()
        assert len(joint_vis.visibility_windows) == 0

    def test_min_vis_keeps_long_enough_windows(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """JointVisibility with min_vis of 0 should keep all windows."""
        joint_vis = JointVisibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[uuid.uuid4(), uuid.uuid4()],
            min_vis=0,
        )
        joint_vis.compute()
        assert len(joint_vis.visibility_windows) > 0


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

    def test_compute_joint_visibility_window_uses_observatory_id_not_instrument_id(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
    ) -> None:
        """Joint window constrained dates should carry observatory IDs from visibilities."""
        instrument_id_1 = uuid.uuid4()
        instrument_id_2 = uuid.uuid4()

        joint_visibility = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[instrument_id_1, instrument_id_2],
        )
        window = joint_visibility.visibility_windows[0].window

        assert (window.begin.observatory_id, window.end.observatory_id) == (
            computed_visibility.observatory_id,
            computed_visibility.observatory_id,
        )

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

    def test_compute_joint_visibility_computed_values_pass_through(
        self, computed_joint_visibility: JointVisibility[EphemerisVisibility]
    ) -> None:
        """JointVisibility should pass through computed_values from input visibilities"""
        # The joint visibility should have computed_values from the first visibility
        # that has them (in this case, both visibilities should have earth_angle)
        assert computed_joint_visibility.computed_values.earth_angle is not None
        assert computed_joint_visibility.computed_values.sun_angle is None
        assert computed_joint_visibility.computed_values.moon_angle is None
        assert computed_joint_visibility.computed_values.alt_az is None

    def test_compute_joint_visibility_min_vis_excludes_short_windows(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """compute_joint_visibility with min_vis larger than all window durations should yield no windows."""
        joint_vis = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
            min_vis=999999,
        )
        assert len(joint_vis.visibility_windows) == 0

    def test_compute_joint_visibility_min_vis_keeps_long_enough_windows(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """compute_joint_visibility with min_vis=0 (default) should keep all windows."""
        joint_vis = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
            min_vis=0,
        )
        assert len(joint_vis.visibility_windows) > 0

    def test_compute_joint_visibility_min_vis_threshold_boundary(
        self,
        computed_visibility: EphemerisVisibility,
        computed_visibility_with_overlap: EphemerisVisibility,
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """Windows exactly at the min_vis boundary should be filtered (duration must be strictly greater)."""
        # Compute with no min_vis to get the actual first window duration.
        baseline = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
            min_vis=0,
        )
        assert len(baseline.visibility_windows) > 0
        window_duration_s = round(
            (
                baseline.visibility_windows[0].window.end.datetime
                - baseline.visibility_windows[0].window.begin.datetime
            ).to_value("s")
        )

        # min_vis equal to the window duration should exclude it (strict >)
        joint_vis_at = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
            min_vis=window_duration_s,
        )
        assert len(joint_vis_at.visibility_windows) == 0

        # min_vis one second less should include it
        joint_vis_below = compute_joint_visibility(
            visibilities=[computed_visibility, computed_visibility_with_overlap],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
            min_vis=window_duration_s - 1,
        )
        assert len(joint_vis_below.visibility_windows) > 0

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

    def test_compute_joint_visibility_handles_end_of_ephemeris_boundary_index(
        self,
        boundary_visibilities: tuple[Visibility, Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """Joint visibility should produce one window at the ephemeris boundary."""
        boundary_joint_visibility = compute_joint_visibility(
            visibilities=[boundary_visibilities[0], boundary_visibilities[1]],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        assert len(boundary_joint_visibility.visibility_windows) == 1

    def test_compute_joint_visibility_boundary_end_constraint_is_window(
        self,
        boundary_visibilities: tuple[Visibility, Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """Joint visibility window at the ephemeris boundary should have end constraint of type WINDOW."""
        boundary_joint_visibility = compute_joint_visibility(
            visibilities=[boundary_visibilities[0], boundary_visibilities[1]],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        boundary_joint_visibility_window = boundary_joint_visibility.visibility_windows[0]
        assert boundary_joint_visibility_window.window.end.constraint == ConstraintType.WINDOW

    def test_compute_joint_visibility_boundary_end_observatory_id_falls_back_to_first(
        self,
        boundary_visibilities: tuple[Visibility, Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
    ) -> None:
        """Joint visibility window at the ephemeris boundary should have end
        observatory_id that falls back to first input observatory_id."""
        boundary_joint_visibility = compute_joint_visibility(
            visibilities=[boundary_visibilities[0], boundary_visibilities[1]],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        boundary_joint_visibility_window = boundary_joint_visibility.visibility_windows[0]
        assert boundary_joint_visibility_window.window.end.observatory_id == test_observatory_id

    def test_compute_joint_visibility_boundary_end_reason_uses_window_fallback(
        self,
        boundary_visibilities: tuple[Visibility, Visibility],
        test_observatory_id: uuid.UUID,
        test_observatory_id_2: uuid.UUID,
        test_observatory_name: str,
    ) -> None:
        """Joint visibility window at the ephemeris boundary should have end
        reason that uses window fallback."""
        boundary_joint_visibility = compute_joint_visibility(
            visibilities=[boundary_visibilities[0], boundary_visibilities[1]],
            instrument_ids=[test_observatory_id, test_observatory_id_2],
        )
        boundary_joint_visibility_window = boundary_joint_visibility.visibility_windows[0]
        assert boundary_joint_visibility_window.constraint_reason.end_reason == (
            f"{test_observatory_name} {ConstraintType.WINDOW.value}"
        )
