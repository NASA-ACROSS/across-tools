import pytest

from across.tools.core.schemas.roll_angle_allowed_range import RollAngleAllowedRange


class TestRollAngleAllowedRangeSchema:
    """Test suite for the RollAngleAllowedRange schema."""

    def test_roll_range_validation(
        self,
    ) -> None:
        """Test initializing RollAngleAllowedRange with invalid range."""
        with pytest.raises(ValueError):
            RollAngleAllowedRange(min_roll=100, max_roll=90)

    def test_roll_step_validation(
        self,
    ) -> None:
        """Test initializing RollAngleAllowedRange with invalid step size."""
        with pytest.raises(ValueError):
            RollAngleAllowedRange(min_roll=10, max_roll=20, step_size=30)
