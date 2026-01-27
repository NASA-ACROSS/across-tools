"""
Logical operators for combining constraints.

This module provides classes to combine multiple constraints using logical operators:
- AndConstraint: Combines constraints with AND logic
- OrConstraint: Combines constraints with OR logic
- NotConstraint: Negates a constraint with NOT logic
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import field_validator

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC

if TYPE_CHECKING:
    from . import Constraint


class ConstraintValidationMixin:
    """
    Mixin to allow ConstraintABC instances to be passed trhough without
    Pydantic validation.

    Provides validators for both single 'constraint' and list 'constraints' fields.
    Validators only apply to fields that exist in the class using this mixin.
    """

    @field_validator("constraints", "constraint", mode="wrap", check_fields=False)
    @classmethod
    def validate_constraints(cls, v: Any, handler: Callable[[Any], Any]) -> Any:
        """Allow both Constraint union types and arbitrary ConstraintABC instances."""
        # For lists of ConstraintABC (including test types), store as-is without validation
        if isinstance(v, ConstraintABC):
            return v
        if isinstance(v, list) and v and isinstance(v[0], ConstraintABC):
            return v
        # For other inputs (dicts from JSON), use normal validation
        return handler(v)


class AndConstraint(ConstraintValidationMixin, ConstraintABC):
    """
    Combines two or more constraints with AND logic.

    Since constraints represent when a target is NOT visible, this constraint is
    violated only if ALL constituent constraints are violated. This is rarely used
    in practice since typical visibility calculations combine constraints with OR
    logic instead. Use AND only when you need ALL specified blocking conditions to
    occur simultaneously (e.g., both Sun AND Moon must be blocking the target).

    The constraints are stored as ConstraintABC instances internally but are serialized
    as part of the Constraint discriminated union for JSON serialization/deserialization.

    Parameters
    ----------
    constraints : list[ConstraintABC]
        List of constraints to combine with AND logic. All constraints must be
        violated for the combined constraint to be violated.

    Examples
    --------
    Create an AND constraint using the & operator (rarely used):

    >>> from across.tools.visibility.constraints import (
    ...     SunAngleConstraint, MoonAngleConstraint, EarthLimbConstraint
    ... )
    >>> sun = SunAngleConstraint(min_angle=45)
    >>> moon = MoonAngleConstraint(min_angle=10)
    >>> earth = EarthLimbConstraint(min_angle=33)
    >>> # Target not visible only when BOTH Sun AND Moon AND Earth all block it
    >>> combined = sun & moon & earth

    Notes
    -----
    For typical visibility calculations, use OR logic (|) instead of AND.
    The | operator combines constraints so that ANY constraint violation blocks visibility,
    which is the standard behavior for combining multiple visibility constraints.
    """

    name: Literal[ConstraintType.AND] = ConstraintType.AND
    short_name: Literal["And"] = "And"
    constraints: list[Constraint]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate AND constraint: returns True only if all sub-constraints are True.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all True
        result = np.ones(len(time), dtype=bool)

        # AND all constraints together
        for constraint in self.constraints:
            result &= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)

        return result


class OrConstraint(ConstraintValidationMixin, ConstraintABC):
    """
    Combines two or more constraints with OR logic.

    Since constraints represent when a target is NOT visible, this constraint is
    violated if ANY constituent constraint is violated. This is the typical and
    recommended way to combine multiple visibility constraints: the target is NOT
    visible if ANY of the constraints block it (Sun OR Moon OR Earth blocking, etc.).

    The constraints are stored as ConstraintABC instances internally but are serialized
    as part of the Constraint discriminated union for JSON serialization/deserialization.

    Parameters
    ----------
    constraints : list[ConstraintABC]
        List of constraints to combine with OR logic. If any constraint is violated,
        the combined constraint is violated.

    Examples
    --------
    Create an OR constraint using the | operator (recommended for typical use):

    >>> from across.tools.visibility.constraints import (
    ...     SunAngleConstraint, MoonAngleConstraint
    ... )
    >>> sun = SunAngleConstraint(min_angle=45)
    >>> moon = MoonAngleConstraint(min_angle=10)
    >>> # Target not visible if Sun OR Moon blocks it
    >>> combined = sun | moon

    More complex example with three constraints:

    >>> from across.tools.visibility.constraints import EarthLimbConstraint
    >>> earth = EarthLimbConstraint(min_angle=33)
    >>> # Target not visible if ANY constraint blocks it
    >>> all_constraints = sun | moon | earth

    Notes
    -----
    OR logic is the standard way to combine constraints. Use this when you want
    observations blocked if ANY of the specified conditions are violated.
    """

    name: Literal[ConstraintType.OR] = ConstraintType.OR
    short_name: Literal["Or"] = "Or"
    constraints: list[Constraint]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate OR constraint: returns True if any sub-constraint is True.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all False
        result = np.zeros(len(time), dtype=bool)

        # OR all constraints together
        for constraint in self.constraints:
            result |= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)

        return result


class NotConstraint(ConstraintValidationMixin, ConstraintABC):
    """
    Negates a constraint with NOT logic.

    Since constraints represent when a target is NOT visible, negating a constraint
    means it is violated when the original constraint is NOT violated. This is useful
    for expressing "target must be in this region" constraints, which act as visibility
    constraints (blocking observations outside that region).

    The constraint is stored as a ConstraintABC instance internally but is serialized
    as part of the Constraint discriminated union for JSON serialization/deserialization.

    Parameters
    ----------
    constraint : ConstraintABC
        The constraint to negate. Can be an atomic constraint or another logical constraint.

    Examples
    --------
    Create a NOT constraint using the ~ operator:

    >>> from across.tools.visibility.constraints import EarthLimbConstraint
    >>> earth = EarthLimbConstraint(min_angle=33)
    >>> # Violated when target is NOT >33° above Earth's limb
    >>> # (i.e., target must stay <33° above limb)
    >>> not_earth = ~earth

    Complex example with combination:

    >>> from across.tools.visibility.constraints import SunAngleConstraint
    >>> sun = SunAngleConstraint(min_angle=45)
    >>> # Target must be >45° from Sun OR must be <33° above Earth limb
    >>> complex_constraint = sun | ~earth

    Notes
    -----
    NOT logic is rarely used in typical visibility calculations. It's primarily
    useful for inverting specific constraints to express regional requirements.
    """

    name: Literal[ConstraintType.NOT] = ConstraintType.NOT
    short_name: Literal["Not"] = "Not"
    constraint: Constraint

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate NOT constraint: returns the negation of the sub-constraint.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        return ~self.constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)


class XorConstraint(ConstraintValidationMixin, ConstraintABC):
    """
    Combines two or more constraints with XOR (exclusive OR) logic.

    Since constraints represent when a target is NOT visible, this constraint is
    violated when an odd number of constituent constraints are violated. For two
    constraints, it's violated when exactly one is violated but not when both are
    violated or neither is violated. This is useful for expressing "either A or B
    but not both" logic.

    The constraints are stored as ConstraintABC instances internally but are serialized
    as part of the Constraint discriminated union for JSON serialization/deserialization.

    Parameters
    ----------
    constraints : list[ConstraintABC]
        List of constraints to combine with XOR logic. The combined constraint is
        violated when an odd number of constituent constraints are violated.

    Examples
    --------
    Create an XOR constraint using the ^ operator:

    >>> from across.tools.visibility.constraints import (
    ...     SunAngleConstraint, MoonAngleConstraint
    ... )
    >>> sun = SunAngleConstraint(min_angle=45)
    >>> moon = MoonAngleConstraint(min_angle=10)
    >>> # Target not visible if exactly one (but not both) blocks it
    >>> combined = sun ^ moon

    More complex example with three constraints:

    >>> from across.tools.visibility.constraints import EarthLimbConstraint
    >>> earth = EarthLimbConstraint(min_angle=33)
    >>> # Violated when odd number of constraints are violated
    >>> all_constraints = sun ^ moon ^ earth

    Notes
    -----
    XOR logic is rarely used in typical visibility calculations but can be useful
    for specific edge cases where you need exclusive conditions.
    """

    name: Literal[ConstraintType.XOR] = ConstraintType.XOR
    short_name: Literal["Xor"] = "Xor"
    constraints: list[Constraint]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate XOR constraint: returns True when odd number of sub-constraints are True.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all False
        result = np.zeros(len(time), dtype=bool)

        # XOR all constraints together
        for constraint in self.constraints:
            result ^= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)

        return result
