import uuid

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility import EphemerisVisibility
from across.tools.visibility.constraints import (
    AndConstraint,
    EarthLimbConstraint,
    NotConstraint,
    OrConstraint,
    XorConstraint,
)


class TestEphemerisVisibility:
    """Test the EphemerisVisibility class."""

    def test_ephemeris_visibility_timestamp(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility computes timestamp."""
        test_visibility.compute()
        assert test_visibility.timestamp is not None

    def test_ephemeris_visibility_windows(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility computes visibility windows."""
        test_visibility.compute()
        assert test_visibility.visibility_windows is not None

    def test_ephemeris_visibility_windows_not_empty(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility computes non-empty visibility windows."""
        test_visibility.compute()
        assert len(test_visibility.visibility_windows) > 0

    def test_ephemeris_visibility_constraints_computed(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility computes constraints."""
        test_visibility.compute()
        assert test_visibility.calculated_constraints is not None

    def test_ephemeris_visibility_earth_constraint(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility includes Earth constraint."""
        test_visibility.compute()
        assert ConstraintType.EARTH in test_visibility.calculated_constraints

    def test_ephemeris_visibility_constraint(self, test_visibility: EphemerisVisibility) -> None:
        """Test that EphemerisVisibility can handle constraints."""

        test_visibility.compute()
        earth_constraints = test_visibility.calculated_constraints[ConstraintType.EARTH]
        first_true_index = int(np.argmax(earth_constraints))

        assert test_visibility._constraint(first_true_index) == ConstraintType.EARTH

    def test_ephemeris_visibility_constraint_fail_no_timestamp(
        self, test_visibility: EphemerisVisibility
    ) -> None:
        """Test that EphemerisVisibility can handle constraints."""

        test_visibility.compute()
        test_visibility.timestamp = None
        earth_constraints = test_visibility.calculated_constraints[ConstraintType.EARTH]
        with pytest.raises(ValueError, match="Timestamp not computed. Call prepare_data\\(\\) first."):
            first_true_index = int(np.argmax(earth_constraints))
            test_visibility._constraint(first_true_index)

    def test_no_constraints(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test EphemerisVisibility with no constraints (lines 91-93)."""
        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=[],  # No constraints
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        vis.compute()
        # With no constraints, inconstraint should be all zeros (nothing violated)
        assert vis.inconstraint is not None
        assert len(vis.inconstraint) == len(vis.timestamp)  # type: ignore[arg-type]
        assert not vis.inconstraint.any()  # All False (no violations)
        # Should still have visibility windows (all time is visible with no constraints)
        assert vis.visibility_windows is not None

    def test_no_constraints_no_timestamp_error(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test EphemerisVisibility with no constraints and no timestamp raises error (line 92)."""
        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=[],  # No constraints
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        # Don't compute timestamp, force prepare_data to be called directly
        # This triggers the error on line 92
        with pytest.raises(ValueError, match="Timestamp not computed. Call _compute_timestamp\\(\\) first."):
            vis.prepare_data()

    def test_compute_timestamp_line_106(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test _compute_timestamp raises error when ephemeris is None (line 106)."""
        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=[],
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        # Set ephemeris to None to trigger line 106 error
        vis.ephemeris = None  # type: ignore[assignment]

        # Call _compute_timestamp - should raise ValueError
        with pytest.raises(ValueError, match="Ephemeris not available for timestamp computation."):
            vis._compute_timestamp()


class TestComputeEphemerisVisibility:
    """Test the compute_ephemeris_visibility function."""

    def test_compute_ephemeris_visibility_returns_correct_type(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that compute_ephemeris_visibility returns an EphemerisVisibility object."""
        assert isinstance(computed_visibility, EphemerisVisibility)

    def test_compute_ephemeris_visibility_observatory_id(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that observatory_id is set correctly."""
        assert computed_visibility.observatory_id is not None


class TestEphemerisVisibilitySingleConstraint:
    """Test that EphemerisVisibility accepts a single constraint (not in a list)."""

    def test_single_constraint_initialization(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_earth_limb_constraint: EarthLimbConstraint,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test that EphemerisVisibility can be initialized with a single constraint."""
        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=test_earth_limb_constraint,  # type: ignore[arg-type]
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        # Verify it's converted to a list internally
        assert isinstance(vis.constraints, list)
        assert len(vis.constraints) == 1
        assert vis.constraints[0] == test_earth_limb_constraint

    def test_single_constraint_compute(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_earth_limb_constraint: EarthLimbConstraint,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test that EphemerisVisibility with single constraint can compute visibility."""
        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=test_earth_limb_constraint,  # type: ignore[arg-type]
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        vis.compute()
        assert vis.visibility_windows is not None
        assert len(vis.visibility_windows) > 0

    def test_single_vs_list_constraint_equivalence(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_earth_limb_constraint: EarthLimbConstraint,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test that single constraint produces same results as list with one constraint."""
        # Create with single constraint
        vis_single = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=test_earth_limb_constraint,  # type: ignore[arg-type]
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        vis_single.compute()

        # Create with list containing single constraint
        vis_list = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=[test_earth_limb_constraint],
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        vis_list.compute()

        # Verify they produce the same results
        assert len(vis_single.visibility_windows) == len(vis_list.visibility_windows)
        assert (
            vis_single.visibility_windows[0].window.begin.datetime
            == vis_list.visibility_windows[0].window.begin.datetime
        )
        assert (
            vis_single.visibility_windows[0].window.end.datetime
            == vis_list.visibility_windows[0].window.end.datetime
        )

    def test_single_combined_constraint(
        self,
        skycoord_near_limb: SkyCoord,
        test_visibility_time_range: tuple[Time, Time],
        test_step_size: TimeDelta,
        test_tle_ephemeris: Ephemeris,
        test_earth_limb_constraint: EarthLimbConstraint,
        test_earth_limb_constraint_2: EarthLimbConstraint,
        test_observatory_name: str,
        test_observatory_id: uuid.UUID,
    ) -> None:
        """Test that a combined constraint (using | operator) works as a single constraint."""
        # Create a combined constraint using | operator
        combined = test_earth_limb_constraint | test_earth_limb_constraint_2

        vis = EphemerisVisibility(
            coordinate=skycoord_near_limb,
            begin=test_visibility_time_range[0],
            end=test_visibility_time_range[1],
            step_size=test_step_size,
            ephemeris=test_tle_ephemeris,
            constraints=combined,  # type: ignore[arg-type]
            observatory_name=test_observatory_name,
            observatory_id=test_observatory_id,
        )
        vis.compute()

        # Verify it works and contains a single OrConstraint in the list
        assert isinstance(vis.constraints, list)
        assert len(vis.constraints) == 1
        assert vis.constraints[0].name == ConstraintType.OR
        assert vis.visibility_windows is not None


class TestFindViolatedConstraint:
    """Test suite for the _find_violated_constraint method."""

    def test_find_violated_constraint_with_no_timestamp(
        self,
        computed_visibility: EphemerisVisibility,
        earth_constraint_33: EarthLimbConstraint,
    ) -> None:
        """Test _find_violated_constraint returns UNKNOWN when timestamp is None (line 133)."""
        # Save original timestamp
        original_timestamp = computed_visibility.timestamp
        # Set timestamp to None
        computed_visibility.timestamp = None

        # Call _find_violated_constraint - should return UNKNOWN
        result = computed_visibility._find_violated_constraint(earth_constraint_33, 0)
        assert result == ConstraintType.UNKNOWN

        # Restore original timestamp
        computed_visibility.timestamp = original_timestamp

    def test_find_violated_constraint_with_no_ephemeris(
        self,
        computed_visibility: EphemerisVisibility,
        earth_constraint_33: EarthLimbConstraint,
    ) -> None:
        """Test _find_violated_constraint returns UNKNOWN when ephemeris is None (line 133)."""
        # Save original ephemeris
        original_ephemeris = computed_visibility.ephemeris
        # Set ephemeris to None
        computed_visibility.ephemeris = None  # type: ignore[assignment]

        # Call _find_violated_constraint - should return UNKNOWN
        result = computed_visibility._find_violated_constraint(earth_constraint_33, 0)
        assert result == ConstraintType.UNKNOWN

        # Restore original ephemeris
        computed_visibility.ephemeris = original_ephemeris

    def test_constraint_when_no_violated_constraints(
        self,
        computed_visibility: EphemerisVisibility,
    ) -> None:
        """Test _constraint returns UNKNOWN when no constraints are violated (line 213)."""
        # Find an index where no constraints are violated
        for idx, is_violated in enumerate(computed_visibility.inconstraint):
            if not is_violated:  # No constraint violated at this index
                result = computed_visibility._constraint(idx)
                # Should return UNKNOWN when no constraints are violated
                assert result == ConstraintType.UNKNOWN
                break

    def test_constraint_with_constraint_type_not_in_constraints_list(
        self,
        computed_visibility: EphemerisVisibility,
    ) -> None:
        """Test _constraint returns constraint_type when not found in self.constraints (line 211)."""
        # Manually add a constraint type to calculated_constraints that doesn't exist in self.constraints
        if computed_visibility.calculated_constraints:
            # Save original constraints
            original_constraints = computed_visibility.constraints

            # Clear constraints list
            computed_visibility.constraints = []

            # Find an index where a constraint is violated
            for idx, is_violated in enumerate(computed_visibility.inconstraint):
                if is_violated:
                    result = computed_visibility._constraint(idx)
                    # Should return the constraint_type string directly since no matching constraint found
                    assert result is not None
                    assert isinstance(result, ConstraintType)
                    break

            # Restore original constraints
            computed_visibility.constraints = original_constraints

    def test_find_violated_constraint_or_no_violations(
        self,
        computed_visibility: EphemerisVisibility,
        or_always_satisfied: OrConstraint,
    ) -> None:
        """Test _find_violated_constraint with OR where no sub-constraints are violated."""

        # Find an index where OR is NOT violated (constraint is satisfied)
        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = or_always_satisfied(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if not result[0]:  # If not violated
                    leaf = computed_visibility._find_violated_constraint(or_always_satisfied, idx)
                    # Should return UNKNOWN when no sub-constraints are violated
                    assert leaf == ConstraintType.UNKNOWN
                    break

    def test_find_violated_constraint_and_no_violations(
        self,
        computed_visibility: EphemerisVisibility,
        and_always_satisfied: AndConstraint,
    ) -> None:
        """Test _find_violated_constraint with AND where no sub-constraints are violated."""

        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = and_always_satisfied(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if not result[0]:  # If not violated
                    leaf = computed_visibility._find_violated_constraint(and_always_satisfied, idx)
                    # Should return UNKNOWN when no sub-constraints are violated
                    assert leaf == ConstraintType.UNKNOWN
                    break

    def test_find_violated_constraint_xor_no_violations(
        self,
        computed_visibility: EphemerisVisibility,
        xor_always_satisfied: XorConstraint,
    ) -> None:
        """Test _find_violated_constraint with XOR where no sub-constraints are violated."""

        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = xor_always_satisfied(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if not result[0]:  # If not violated (even number of violations)
                    leaf = computed_visibility._find_violated_constraint(xor_always_satisfied, idx)
                    # Should return UNKNOWN when no sub-constraints are violated
                    assert leaf == ConstraintType.UNKNOWN
                    break

    def test_find_violated_constraint_xor_return_unknown_line_173(
        self,
        computed_visibility: EphemerisVisibility,
        xor_sun_earth: XorConstraint,
    ) -> None:
        """Test _find_violated_constraint XOR recursive call (line 173)."""
        # Find an index where XOR IS violated (odd number of violations)
        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = xor_sun_earth(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if result[0]:  # If XOR IS violated
                    # This should hit line 173 (recursive call in XOR branch)
                    leaf = computed_visibility._find_violated_constraint(xor_sun_earth, idx)
                    assert leaf in [ConstraintType.SUN, ConstraintType.EARTH]
                    break

    def test_find_violated_constraint_leaf(
        self,
        computed_visibility: EphemerisVisibility,
        earth_constraint_33: EarthLimbConstraint,
    ) -> None:
        """Test _find_violated_constraint with a regular leaf constraint directly."""

        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = earth_constraint_33(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if result[0]:  # If violated
                    # Calling _find_violated_constraint on a leaf should return its name
                    leaf = computed_visibility._find_violated_constraint(earth_constraint_33, idx)
                    assert leaf == ConstraintType.EARTH
                    break

    def test_find_violated_constraint_nested_not_or(
        self,
        computed_visibility: EphemerisVisibility,
        not_or_sun_earth: NotConstraint,
    ) -> None:
        """Test _find_violated_constraint with NOT(OR(...)) nested constraint."""

        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = not_or_sun_earth(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if result[0]:  # If violated
                    leaf = computed_visibility._find_violated_constraint(not_or_sun_earth, idx)
                    # Should drill through NOT to the OR and find a leaf
                    assert leaf in [ConstraintType.SUN, ConstraintType.EARTH, ConstraintType.UNKNOWN]
                    break

    def test_find_violated_constraint_deeply_nested(
        self,
        computed_visibility: EphemerisVisibility,
        and_or_sun_earth: AndConstraint,
    ) -> None:
        """Test _find_violated_constraint with deeply nested logical constraints."""

        if computed_visibility.timestamp is not None:
            for idx in range(len(computed_visibility.timestamp)):
                result = and_or_sun_earth(
                    time=computed_visibility.timestamp[idx : idx + 1],
                    ephemeris=computed_visibility.ephemeris,
                    coordinate=computed_visibility.coordinate,
                )
                if result[0]:  # If violated
                    leaf = computed_visibility._find_violated_constraint(and_or_sun_earth, idx)
                    # Should drill down through AND to OR and find a leaf
                    assert leaf in [ConstraintType.SUN, ConstraintType.EARTH, ConstraintType.UNKNOWN]
                    break
