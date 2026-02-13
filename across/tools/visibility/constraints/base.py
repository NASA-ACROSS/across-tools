from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...core.schemas.base import BaseSchema
from ...core.schemas.visibility import VisibilityComputedValues
from ...ephemeris import Ephemeris


def get_slice(time: Time, ephemeris: Ephemeris) -> slice:
    """
    Return a slice for what the part of the ephemeris that we're using.

    Arguments
    ---------
    time : Time
        The time to calculate the slice for
    ephemeris : Ephemeris
        The spacecraft ephemeris

    Returns
    -------
        The slice for the ephemeris
    """
    # Ensure that time is an array-like object
    if time.isscalar:
        raise NotImplementedError("Scalar time not supported")

    # Find the indices for the start and end of the time range and return a
    # slice for that range
    return slice(ephemeris.index(time[0]), ephemeris.index(time[-1]) + 1)


class ConstraintABC(BaseSchema, ABC):
    """
    Base class for constraints. Constraints represent conditions that make a target
    NOT visible (blocked from observation). A constraint returns True when the target
    is constrained (NOT visible) and False when the target is visible.

    When combining multiple constraints, the typical approach in visibility calculations
    is to use OR logic: the target is not visible if ANY constraint blocks it. AND and
    NOT logic are provided but are rarely used in practice.

    Constraints can be combined using logical operators:
    - | (OR): Combined constraint violated if ANY sub-constraint is violated (typical usage)
    - & (AND): Combined constraint violated only if ALL sub-constraints are violated (rarely used)
    - ~ (NOT): Inverts the constraint's logic (rarely used)
    - ^ (XOR): Combined constraint violated when an odd number of sub-constraints are violated (rarely used)

    Methods
    -------
    __call__(time, ephemeris, coord)
        Evaluates if a given coordinate is constrained at the given time.
    __or__(other)
        Combines two constraints with OR logic using the | operator (typical usage).
    __and__(other)
        Combines two constraints with AND logic using the & operator.
    __invert__()
        Negates a constraint using the ~ operator.
    __xor__(other)
        Combines two constraints with XOR logic using the ^ operator.

    Examples
    --------
    Typical usage: combine constraints with OR (if any constraint blocks, target not visible):

    >>> from across.tools.visibility.constraints import (
    ...     SunAngleConstraint, MoonAngleConstraint, EarthLimbConstraint
    ... )
    >>> sun = SunAngleConstraint(min_angle=45)      # Not visible if <45° from Sun
    >>> moon = MoonAngleConstraint(min_angle=10)    # Not visible if <10° from Moon
    >>> earth = EarthLimbConstraint(min_angle=33)   # Not visible if <33° above limb
    >>> # Target not visible if ANY constraint blocks it
    >>> all_constraints = sun | moon | earth
    """

    short_name: str
    name: ConstraintType
    computed_values: VisibilityComputedValues = Field(default_factory=VisibilityComputedValues, exclude=True)

    @abstractmethod
    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the constraint.

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
            Boolean array where True indicates the coordinate violates the constraint
            (is not visible).
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    def __and__(self, other: ConstraintABC) -> ConstraintABC:
        """
        Combine two constraints with AND logic using the & operator.

        This creates a composite constraint that is violated only when BOTH
        component constraints are violated. This is rarely used in practice since
        the typical way to combine visibility constraints is with OR logic (use | instead).

        Parameters
        ----------
        other : ConstraintABC
            The constraint to combine with this one

        Returns
        -------
        AndConstraint
            A new constraint that is violated only if both constraints are violated

        Examples
        --------
        >>> from across.tools.visibility.constraints import (
        ...     SunAngleConstraint, MoonAngleConstraint
        ... )
        >>> sun = SunAngleConstraint(min_angle=45)
        >>> moon = MoonAngleConstraint(min_angle=10)
        >>> # Rarely used: target not visible only if BOTH block it
        >>> combined = sun & moon
        """
        # Import here to avoid circular imports
        from .logical import AndConstraint

        return AndConstraint(constraints=[self, other])

    def __or__(self, other: ConstraintABC) -> ConstraintABC:
        """
        Combine two constraints with OR logic using the | operator.

        This creates a composite constraint that is violated when ANY component
        constraint is violated. This is the typical and recommended way to combine
        visibility constraints: the target is not visible if ANY constraint blocks it.

        Parameters
        ----------
        other : ConstraintABC
            The constraint to combine with this one

        Returns
        -------
        OrConstraint
            A new constraint that is violated if either constraint is violated

        Examples
        --------
        >>> from across.tools.visibility.constraints import (
        ...     SunAngleConstraint, MoonAngleConstraint
        ... )
        >>> sun = SunAngleConstraint(min_angle=45)
        >>> moon = MoonAngleConstraint(min_angle=10)
        >>> # Typical usage: target not visible if Sun OR Moon blocks it
        >>> combined = sun | moon
        """
        # Import here to avoid circular imports
        from .logical import OrConstraint

        return OrConstraint(constraints=[self, other])

    def __invert__(self) -> ConstraintABC:
        """
        Negate a constraint using the ~ operator.

        This creates a composite constraint with inverted logic: where the original
        constraint is violated (target not visible), the negated constraint is not violated
        (target visible), and vice versa. This is rarely used in practice.

        Returns
        -------
        NotConstraint
            A new constraint that is the logical negation of this constraint

        Examples
        --------
        >>> from across.tools.visibility.constraints import EarthLimbConstraint
        >>> earth = EarthLimbConstraint(min_angle=33)
        >>> # Violated when earth constraint is NOT violated
        >>> not_earth = ~earth
        """
        # Import here to avoid circular imports
        from .logical import NotConstraint

        return NotConstraint(constraint=self)

    def __xor__(self, other: ConstraintABC) -> ConstraintABC:
        """
        Combine two constraints with XOR logic using the ^ operator.

        This creates a composite constraint that is violated when exactly one (but not both)
        of the component constraints is violated. For multiple constraints, it follows
        standard XOR behavior: violated when an odd number of constraints are violated.
        This is rarely used in practice.

        Parameters
        ----------
        other : ConstraintABC
            The constraint to combine with this one

        Returns
        -------
        XorConstraint
            A new constraint that is violated when an odd number of constraints are violated

        Examples
        --------
        >>> from across.tools.visibility.constraints import (
        ...     SunAngleConstraint, MoonAngleConstraint
        ... )
        >>> sun = SunAngleConstraint(min_angle=45)
        >>> moon = MoonAngleConstraint(min_angle=10)
        >>> # Rarely used: target not visible if exactly one (but not both) blocks it
        >>> combined = sun ^ moon
        """
        # Import here to avoid circular imports
        from .logical import XorConstraint

        return XorConstraint(constraints=[self, other])
