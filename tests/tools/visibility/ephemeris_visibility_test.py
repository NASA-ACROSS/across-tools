from typing import Any

import numpy as np
import pytest

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.visibility import EphemerisVisibility


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

    def test_merge_computed_values_sun(
        self, test_visibility: EphemerisVisibility, mock_constraint_class: Any
    ) -> None:
        """Test _merge_computed_values with Sun constraint."""
        mock_sun_constraint = mock_constraint_class(ConstraintType.SUN, "sun_angle")
        test_visibility.constraints = [mock_sun_constraint]
        test_visibility._merge_computed_values()

        assert test_visibility.computed_values.sun_angle == mock_sun_constraint.computed_values.sun_angle

    def test_merge_computed_values_moon(
        self, test_visibility: EphemerisVisibility, mock_constraint_class: Any
    ) -> None:
        """Test _merge_computed_values with Moon constraint."""
        mock_moon_constraint = mock_constraint_class(ConstraintType.MOON, "moon_angle")
        test_visibility.constraints = [mock_moon_constraint]
        test_visibility._merge_computed_values()

        assert test_visibility.computed_values.moon_angle == mock_moon_constraint.computed_values.moon_angle

    def test_merge_computed_values_alt_az(
        self,
        test_visibility: EphemerisVisibility,
        mock_constraint_class: Any,
    ) -> None:
        """Test _merge_computed_values with AltAz constraint."""
        mock_alt_az_constraint = mock_constraint_class(ConstraintType.ALT_AZ, "alt_az")
        test_visibility.constraints = [mock_alt_az_constraint]
        test_visibility._merge_computed_values()

        assert test_visibility.computed_values.alt_az == mock_alt_az_constraint.computed_values.alt_az


class TestComputeEphemerisVisibility:
    """Test the compute_ephemeris_visibility function."""

    def test_compute_ephemeris_visibility_returns_correct_type(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that compute_ephemeris_visibility returns an EphemerisVisibility object."""
        assert isinstance(computed_visibility, EphemerisVisibility)

    def test_compute_ephemeris_visibility_windows_not_none(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that visibility_windows is not None."""
        assert computed_visibility.visibility_windows is not None

    def test_compute_ephemeris_visibility_windows_not_empty(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that visibility_windows is not empty."""
        assert len(computed_visibility.visibility_windows) > 0

    def test_compute_ephemeris_visibility_constraints_not_none(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that calculated_constraints is not None."""
        assert computed_visibility.calculated_constraints is not None

    def test_compute_ephemeris_visibility_observatory_id(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that observatory_id is set correctly."""
        assert computed_visibility.observatory_id is not None

    def test_compute_ephemeris_visibility_earth_constraint(
        self, computed_visibility: EphemerisVisibility
    ) -> None:
        """Test that Earth constraint is included in calculated_constraints."""
        assert ConstraintType.EARTH in computed_visibility.calculated_constraints
