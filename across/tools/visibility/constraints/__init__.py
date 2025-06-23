from typing import Annotated

from pydantic import Field

from .earth_limb import EarthLimbConstraint
from .moon_angle import MoonAngleConstraint
from .sun_angle import SunAngleConstraint

__all__ = [
    "Constraint",
    "get_slice",
    "EarthLimbConstraint",
    "MoonAngleConstraint",
    "SunAngleConstraint",
]

# Define a type that covers all constraints
Constraint = Annotated[
    EarthLimbConstraint | MoonAngleConstraint | SunAngleConstraint,
    Field(discriminator="name"),
]
