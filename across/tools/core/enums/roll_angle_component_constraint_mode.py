from enum import Enum


class ComponentConstraintMode(str, Enum):
    """
    How per-component constraint violations combine into a target constraint.

    ANY: target is constrained if any component vector violates its constraints.
    ALL: target is constrained only if all component vectors violate their constraints.
    """

    ANY = "any"
    ALL = "all"
