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

from pydantic import TypeAdapter

from .constraints import Constraint

Constraints = TypeAdapter(list[Constraint])


def constraints_from_json(input: str) -> list[Constraint]:
    """
    Load constraints from a JSON string.

    Parameters
    ----------
    input : str
        JSON string containing the constraints.

    Returns
    -------
    list[Constraint]
        List of Constraint objects loaded from the JSON string.
    """
    return Constraints.validate_json(input)


def constraints_to_json(constraints: list[Constraint]) -> str:
    """
    Convert constraints to a JSON string.

    Parameters
    ----------
    constraints : list[Constraint]
        List of Constraint objects to convert.

    Returns
    -------
    str
        JSON string representation of the constraints.
    """
    return Constraints.dump_json(constraints, exclude_none=True).decode()
