"""
Constraint JSON serialization and deserialization utilities.

This module provides functions for converting between Constraint objects and their
JSON string representations. It handles the serialization and deserialization
of constraint data for storage, transmission, or configuration purposes.

Functions
---------
constraint_from_json : function
    Load constraints from a JSON string into Constraint objects.
constraint_to_json : function
    Convert Constraint objects to a JSON string representation.

Dependencies
------------
- json : Standard library JSON handling
- .constraints : Local module containing Constraint and Constraints classes

Examples
--------
>>> constraints_json = '[{"short_name": "Sun", "name": "Sun Angle", "min_angle": 45.0}]'
>>> constraints = constraint_from_json(constraints_json)
>>> json_output = constraint_to_json(constraints)
"""

from typing import Any

from pydantic import BaseModel, TypeAdapter, model_serializer, model_validator

from .constraints import AllConstraint

Constraints: TypeAdapter[list[AllConstraint]] = TypeAdapter(list[AllConstraint])


class Constraint(BaseModel):
    """Model for deserializing a single constraint."""

    root: AllConstraint

    @model_validator(mode="before")
    @classmethod
    def unwrap_input(cls, data: dict[str, Any] | Any) -> dict[str, dict[str, Any]] | Any:
        """Unwrap input data to extract constraint."""
        # If input is already parsed, extract constraint
        if isinstance(data, dict) and "root" not in data:
            return {"root": data}
        return data

    @model_serializer
    def serialize_constraint(self) -> dict[str, Any]:
        """Serialize the constraint to a dictionary."""
        return self.root.model_dump(exclude_none=True)


def constraints_from_json(input: str) -> AllConstraint | list[AllConstraint]:
    """
    Load constraints from a JSON string.

    Parameters
    ----------
    input : str
        JSON string containing the constraints.

    Returns
    -------
    AllConstraint | list[AllConstraint]
        Single Constraint object or list of Constraint objects loaded from the JSON string.
    """
    # Try to parse as a single constraint first
    try:
        model = Constraint.model_validate_json(input)
        return model.root
    except Exception:
        # If that fails, try to parse as a list
        return Constraints.validate_json(input)


def constraints_to_json(constraints: AllConstraint | list[AllConstraint]) -> str:
    """
    Convert constraints to a JSON string.

    Parameters
    ----------
    constraints : AllConstraint | list[AllConstraint]
        Single Constraint object or list of Constraint objects to convert.

    Returns
    -------
    str
        JSON string representation of the constraints.
    """
    if isinstance(constraints, list):
        return Constraints.dump_json(constraints, exclude_none=True).decode()
    else:
        return Constraint(root=constraints).model_dump_json(exclude_none=True)
