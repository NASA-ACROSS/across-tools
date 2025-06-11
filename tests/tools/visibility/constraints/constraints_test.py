from unittest.mock import MagicMock

import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.base import ConstraintABC, get_slice


class TestConstraintABC:
    """Test class for ConstraintABC functionality."""

    def test_constraint_is_instance_of_constraint(self, dummy_constraint: ConstraintABC) -> None:
        """Test that ConstraintABC is instance of ConstraintABC."""
        assert isinstance(dummy_constraint, ConstraintABC)

    def test_constraint_short_name_value(self, dummy_constraint: ConstraintABC) -> None:
        """Test short_name value is set correctly."""
        assert dummy_constraint.name == ConstraintType.UNKNOWN

    def test_constraint_name_value(self, dummy_constraint: ConstraintABC) -> None:
        """Test name value is set correctly."""
        assert dummy_constraint.short_name == "Dummy"

    def test_get_slice_raises_for_scalar_time(self, scalar_time: Time) -> None:
        """Test get_slice raises NotImplementedError for scalar time."""
        mock_ephemeris = MagicMock()
        with pytest.raises(NotImplementedError):
            get_slice(scalar_time, mock_ephemeris)


class TestGetSlice:
    """Test suite for get_slice function."""

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
