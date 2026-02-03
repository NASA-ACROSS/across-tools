import json

from across.tools.visibility import constraints_from_json, constraints_to_json
from across.tools.visibility.constraints import AllConstraint
from across.tools.visibility.constraints.base import ConstraintABC
from across.tools.visibility.constraints.moon_angle import MoonAngleConstraint
from across.tools.visibility.constraints.sun_angle import SunAngleConstraint


class TestConstraintConstructor:
    """Test suite for the constraint constructor."""

    def test_constraints_from_json_returns_list(self, constraint_json: str) -> None:
        """Test that constraints_from_json returns a list."""
        constraints = constraints_from_json(constraint_json)
        assert isinstance(constraints, list)

    def test_constraints_from_json_returns_constraint_objects(self, constraint_json: str) -> None:
        """Test that all items in the list are ConstraintABC instances."""
        constraints = constraints_from_json(constraint_json)
        assert isinstance(constraints, list)  # Ensure it's a list for this test
        assert all(isinstance(c, ConstraintABC) for c in constraints)

    def test_constraints_from_json_returns_correct_count(self, constraint_json: str) -> None:
        """Test that the correct number of constraints are loaded."""
        constraints = constraints_from_json(constraint_json)
        assert isinstance(constraints, list)  # Ensure it's a list for this test
        assert len(constraints) == 3

    def test_constraints_to_json_returns_string(self, constraints_from_fixture: list[AllConstraint]) -> None:
        """Test that constraints_to_json returns a string."""
        json_output = constraints_to_json(constraints_from_fixture)
        assert isinstance(json_output, str)

    def test_constraints_to_json_preserves_data(
        self, constraint_json: str, constraints_from_fixture: list[AllConstraint]
    ) -> None:
        """Test that converting to JSON preserves the original data."""
        json_output = constraints_to_json(constraints_from_fixture)
        assert json.loads(json_output) == json.loads(constraint_json)

    def test_constraints_from_json_single_constraint(self) -> None:
        """Test that constraints_from_json handles a single constraint JSON."""
        single_constraint_json = '{"short_name": "Sun", "name": "Sun Angle", "min_angle": 45.0}'
        constraint = constraints_from_json(single_constraint_json)
        assert isinstance(constraint, ConstraintABC)

    def test_constraints_to_json_single_constraint(self) -> None:
        """Test that constraints_to_json handles a single constraint."""
        single_constraint = SunAngleConstraint(min_angle=45.0)
        json_output = constraints_to_json(single_constraint)
        assert isinstance(json_output, str)
        # Verify it can be parsed back
        parsed = json.loads(json_output)
        assert parsed["short_name"] == "Sun"

    def test_constraints_from_json_list_constraint(self) -> None:
        """Test that constraints_from_json handles a list of constraints JSON."""
        list_constraint_json = (
            '[{"short_name": "Sun", "name": "Sun Angle", "min_angle": 45.0},'
            ' {"short_name": "Moon", "name": "Moon Angle", "max_angle": 30.0}]'
        )
        constraints = constraints_from_json(list_constraint_json)
        assert isinstance(constraints, list)
        assert len(constraints) == 2
        assert all(isinstance(c, ConstraintABC) for c in constraints)

    def test_constraints_to_json_list_constraint(self) -> None:
        """Test that constraints_to_json handles a list of constraints."""
        list_constraints: list[AllConstraint] = [
            SunAngleConstraint(min_angle=45.0),
            MoonAngleConstraint(max_angle=30.0),
        ]
        json_output = constraints_to_json(list_constraints)
        assert isinstance(json_output, str)
        # Verify it can be parsed back
        parsed = json.loads(json_output)
        assert len(parsed) == 2
        assert parsed[0]["short_name"] == "Sun"
