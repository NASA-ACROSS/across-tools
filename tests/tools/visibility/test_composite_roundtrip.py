"""
Test suite for round-trip serialization of composite constraints.

Tests that composite constraints (logical operators like AND, OR, NOT) can be
serialized to JSON and deserialized back with all data intact.
"""

import json
from typing import cast

from across.tools.visibility import constraints_from_json, constraints_to_json
from across.tools.visibility.constraints import (
    AndConstraint,
    MoonAngleConstraint,
    NotConstraint,
    OrConstraint,
    SunAngleConstraint,
    XorConstraint,
)


class TestCompositeRoundTrip:
    """Test suite for round-trip serialization of composite constraints."""

    def test_and_constraint_roundtrip(self) -> None:
        """Test that AND constraint serializes and deserializes correctly."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        composite: AndConstraint = cast(AndConstraint, sun & moon)

        # Serialize to JSON
        json_str = constraints_to_json(composite)
        assert isinstance(json_str, str)

        # Deserialize back
        restored = constraints_from_json(json_str)
        assert isinstance(restored, AndConstraint)
        assert restored == composite

    def test_or_constraint_roundtrip(self) -> None:
        """Test that OR constraint serializes and deserializes correctly."""
        sun = SunAngleConstraint(max_angle=30)
        moon = MoonAngleConstraint(max_angle=20)
        composite: OrConstraint = cast(OrConstraint, sun | moon)

        # Serialize to JSON
        json_str = constraints_to_json(composite)
        assert isinstance(json_str, str)

        # Deserialize back
        restored = constraints_from_json(json_str)
        assert isinstance(restored, OrConstraint)
        assert restored == composite

    def test_not_constraint_roundtrip(self) -> None:
        """Test that NOT constraint serializes and deserializes correctly."""
        sun = SunAngleConstraint(min_angle=45)
        negated: NotConstraint = cast(NotConstraint, ~sun)

        # Serialize to JSON
        json_str = constraints_to_json(negated)
        assert isinstance(json_str, str)

        # Deserialize back
        restored = constraints_from_json(json_str)
        assert isinstance(restored, NotConstraint)
        assert restored == negated

    def test_xor_constraint_roundtrip(self) -> None:
        """Test that XOR constraint serializes and deserializes correctly."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        composite: XorConstraint = cast(XorConstraint, sun ^ moon)

        # Serialize to JSON
        json_str = constraints_to_json(composite)
        assert isinstance(json_str, str)

        # Deserialize back
        restored = constraints_from_json(json_str)
        assert isinstance(restored, XorConstraint)
        assert restored == composite

    def test_nested_composite_constraint_roundtrip(self) -> None:
        """Test that nested composite constraints serialize and deserialize correctly."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)

        # Create nested constraint: (sun & moon) | ~sun
        inner: AndConstraint = cast(AndConstraint, sun & moon)
        outer: OrConstraint = cast(OrConstraint, inner | (~sun))

        # Serialize to JSON
        json_str = constraints_to_json(outer)
        assert isinstance(json_str, str)

        # Deserialize back
        restored = constraints_from_json(json_str)
        assert restored == outer

    def test_constraint_parameters_preserved_in_and(self) -> None:
        """Test that constraint parameters are preserved in AND constraint roundtrip."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10, max_angle=30)
        composite: AndConstraint = cast(AndConstraint, sun & moon)

        # Serialize and deserialize
        json_str = constraints_to_json(composite)
        restored = cast(AndConstraint, constraints_from_json(json_str))

        # Verify the restored constraints have the same parameters
        assert isinstance(restored, AndConstraint)
        restored_constraints = restored.constraints

        # Check first constraint (Sun)
        assert restored_constraints[0].short_name == "Sun"
        sun_constraint = cast(SunAngleConstraint, restored_constraints[0])
        assert sun_constraint.min_angle == 45.0

        # Check second constraint (Moon)
        assert restored_constraints[1].short_name == "Moon"
        moon_constraint = cast(MoonAngleConstraint, restored_constraints[1])
        assert moon_constraint.min_angle == 10.0
        assert moon_constraint.max_angle == 30.0

    def test_constraint_parameters_preserved_in_or(self) -> None:
        """Test that constraint parameters are preserved in OR constraint roundtrip."""
        sun = SunAngleConstraint(max_angle=30)
        moon = MoonAngleConstraint(min_angle=15, max_angle=25)
        composite: OrConstraint = cast(OrConstraint, sun | moon)

        # Serialize and deserialize
        json_str = constraints_to_json(composite)
        restored = cast(OrConstraint, constraints_from_json(json_str))

        # Verify the restored constraints have the same parameters
        assert isinstance(restored, OrConstraint)
        restored_constraints = restored.constraints

        # Check first constraint (Sun)
        assert restored_constraints[0].short_name == "Sun"
        sun_constraint = cast(SunAngleConstraint, restored_constraints[0])
        assert sun_constraint.max_angle == 30.0

        # Check second constraint (Moon)
        assert restored_constraints[1].short_name == "Moon"
        moon_constraint = cast(MoonAngleConstraint, restored_constraints[1])
        assert moon_constraint.min_angle == 15.0
        assert moon_constraint.max_angle == 25.0

    def test_json_preserves_constraint_structure(self) -> None:
        """Test that JSON output includes constraint parameters."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        composite: AndConstraint = cast(AndConstraint, sun & moon)

        # Serialize to JSON
        json_str = constraints_to_json(composite)
        json_obj = json.loads(json_str)

        # Verify JSON structure includes constraint parameters
        assert json_obj["short_name"] == "And"
        assert len(json_obj["constraints"]) == 2

        # Check that each constraint in the JSON has its parameters
        assert json_obj["constraints"][0]["short_name"] == "Sun"
        assert json_obj["constraints"][0].get("min_angle") == 45.0

        assert json_obj["constraints"][1]["short_name"] == "Moon"
        assert json_obj["constraints"][1].get("min_angle") == 10.0
