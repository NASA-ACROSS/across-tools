import json

from across.tools.visibility import constraints_from_json, constraints_to_json
from across.tools.visibility.constraints import Constraint
from across.tools.visibility.constraints.base import ConstraintABC


class TestConstraintConstructor:
    """Test suite for the constraint constructor."""

    def test_constraints_from_json_returns_list(self, constraint_json: str) -> None:
        """Test that constraints_from_json returns a list."""
        constraints = constraints_from_json(constraint_json)
        assert isinstance(constraints, list)

    def test_constraints_from_json_returns_constraint_objects(self, constraint_json: str) -> None:
        """Test that all items in the list are ConstraintABC instances."""
        constraints = constraints_from_json(constraint_json)
        assert all(isinstance(c, ConstraintABC) for c in constraints)

    def test_constraints_from_json_returns_correct_count(self, constraint_json: str) -> None:
        """Test that the correct number of constraints are loaded."""
        constraints = constraints_from_json(constraint_json)
        assert len(constraints) == 3

    def test_constraints_to_json_returns_string(self, constraints_from_fixture: list[Constraint]) -> None:
        """Test that constraints_to_json returns a string."""
        json_output = constraints_to_json(constraints_from_fixture)
        assert isinstance(json_output, str)

    def test_constraints_to_json_preserves_data(
        self, constraint_json: str, constraints_from_fixture: list[Constraint]
    ) -> None:
        """Test that converting to JSON preserves the original data."""
        json_output = constraints_to_json(constraints_from_fixture)
        assert json.loads(json_output) == json.loads(constraint_json)
