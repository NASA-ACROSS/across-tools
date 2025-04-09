from unittest.mock import MagicMock

import pytest
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.core.schemas.visibility import Constraint
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.base import get_slice


class TestConstraint:
    """Test class for Constraint functionality."""

    def test_constraint_is_instance_of_constraint(self, dummy_constraint: Constraint) -> None:
        """Test that Constraint is instance of Constraint."""
        assert isinstance(dummy_constraint, Constraint)

    def test_constraint_min_angle_value(self, dummy_constraint_with_angles: Constraint) -> None:
        """Test min_angle value is set correctly."""
        assert dummy_constraint_with_angles.min_angle == 10

    def test_constraint_max_angle_value(self, dummy_constraint_with_angles: Constraint) -> None:
        """Test max_angle value is set correctly."""
        assert dummy_constraint_with_angles.max_angle == 20

    def test_constraint_short_name_value(self, dummy_constraint: Constraint) -> None:
        """Test short_name value is set correctly."""
        assert dummy_constraint.short_name == ConstraintType.UNKNOWN

    def test_constraint_name_value(self, dummy_constraint: Constraint) -> None:
        """Test name value is set correctly."""
        assert dummy_constraint.name == "Dummy Constraint"

    def test_constraint_optional_min_angle_is_none(self, dummy_constraint: Constraint) -> None:
        """Test min_angle is None by default."""
        assert dummy_constraint.min_angle is None

    def test_constraint_optional_max_angle_is_none(self, dummy_constraint: Constraint) -> None:
        """Test max_angle is None by default."""
        assert dummy_constraint.max_angle is None

    def test_get_slice_raises_for_scalar_time(self, scalar_time: Time) -> None:
        """Test get_slice raises NotImplementedError for scalar time."""
        mock_ephemeris = MagicMock()
        with pytest.raises(NotImplementedError):
            get_slice(scalar_time, mock_ephemeris)

    def test_constraint_validation_min_angle_invalid(self, dummy_constraint_class: type[Constraint]) -> None:
        """Test validation error for invalid min_angle type."""
        with pytest.raises(ValidationError):
            dummy_constraint_class(short_name="dummy", name="Dummy Constraint", min_angle="invalid")  # type: ignore[arg-type]

    def test_constraint_validation_max_angle_invalid(self, dummy_constraint_class: type[Constraint]) -> None:
        """Test validation error for invalid max_angle type."""
        with pytest.raises(ValidationError):
            dummy_constraint_class(short_name="dummy", name="Dummy Constraint", max_angle="invalid")  # type: ignore[arg-type]

    def test_get_slice_time_array_start(self, time_array: Time, mock_ephemeris: Ephemeris) -> None:
        """Test get_slice start index is correct."""
        result = get_slice(time_array, mock_ephemeris)
        assert result.start == 0

    def test_get_slice_time_array_stop(self, time_array: Time, mock_ephemeris: Ephemeris) -> None:
        """Test get_slice stop index is correct."""
        result = get_slice(time_array, mock_ephemeris)
        assert result.stop == 1441

    def test_get_slice_time_array_type(self, time_array: Time, mock_ephemeris: Ephemeris) -> None:
        """Test get_slice returns slice object."""
        result = get_slice(time_array, mock_ephemeris)
        assert isinstance(result, slice)
