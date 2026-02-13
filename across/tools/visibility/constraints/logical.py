"""
Logical operators for combining constraints.

This module provides classes to combine multiple constraints using logical operators:
- AndConstraint: Combines constraints with AND logic
- OrConstraint: Combines constraints with OR logic
- NotConstraint: Negates a constraint with NOT logic
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import TypeAdapter, field_validator, model_serializer

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC

if TYPE_CHECKING:
    from . import AllConstraint


@lru_cache(maxsize=1)
def _constraint_adapter() -> TypeAdapter[AllConstraint]:
    from . import AllConstraint

    return TypeAdapter(AllConstraint)


@lru_cache(maxsize=1)
def _constraints_adapter() -> TypeAdapter[list[AllConstraint]]:
    from . import AllConstraint

    return TypeAdapter(list[AllConstraint])


class ConstraintCoercionMixin:
    """
    Coerce constraint inputs into ConstraintABC instances.

    Accepts either already-instantiated ConstraintABC objects or JSON-like dicts
    that are parsed through the Constraint discriminated union.
    """

    @field_validator("constraint", mode="before", check_fields=False)
    @classmethod
    def validate_constraint(cls, v: Any) -> Any:
        """
        Coerce single constraint input into ConstraintABC instance.
        """
        if isinstance(v, ConstraintABC):
            return v
        return _constraint_adapter().validate_python(v)

    @field_validator("constraints", mode="before", check_fields=False)
    @classmethod
    def validate_constraints(cls, v: Any) -> Any:
        """
        Coerce list of constraints input into list of ConstraintABC instances.
        """
        if isinstance(v, list) and all(isinstance(item, ConstraintABC) for item in v):
            return v
        return _constraints_adapter().validate_python(v)

    @model_serializer
    def serialize_logical_constraint(self) -> dict[str, Any]:
        """Serialize logical constraint with full constraint details."""
        result: dict[str, Any] = {
            "short_name": self.short_name,  # type: ignore[attr-defined]
            "name": self.name.value,  # type: ignore[attr-defined]
        }

        # Handle constraints field for And/Or/Xor
        if hasattr(self, "constraints"):
            result["constraints"] = [c.model_dump(exclude_none=True) for c in self.constraints]
        # Handle constraint field for Not
        elif hasattr(self, "constraint"):
            result["constraint"] = self.constraint.model_dump(exclude_none=True)

        return result


class AndConstraint(ConstraintCoercionMixin, ConstraintABC):
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
    constraints: list[ConstraintABC]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
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
        npt.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all True
        result = np.ones(len(time), dtype=bool)

        # AND all constraints together
        for constraint in self.constraints:
            result &= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)
            self.computed_values.merge(constraint.computed_values)

        return result


class OrConstraint(ConstraintCoercionMixin, ConstraintABC):
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
    constraints: list[ConstraintABC]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
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
        npt.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all False
        result = np.zeros(len(time), dtype=bool)

        # OR all constraints together
        for constraint in self.constraints:
            result |= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)
            self.computed_values.merge(constraint.computed_values)

        return result


class NotConstraint(ConstraintCoercionMixin, ConstraintABC):
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
    constraint: ConstraintABC

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
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
        npt.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        constraint = ~self.constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)
        self.computed_values.merge(self.constraint.computed_values)
        return constraint


class XorConstraint(ConstraintCoercionMixin, ConstraintABC):
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
    constraints: list[ConstraintABC]

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
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
        npt.NDArray[np.bool_]
            Boolean array where True indicates constraint is violated.
        """
        if not self.constraints:
            return np.zeros(len(time), dtype=bool)

        # Start with all False
        result = np.zeros(len(time), dtype=bool)

        # XOR all constraints together
        for constraint in self.constraints:
            result ^= constraint(time=time, ephemeris=ephemeris, coordinate=coordinate)
            self.computed_values.merge(constraint.computed_values)

        return result
