from typing import Annotated

from pydantic import Field

from .alt_az import AltAzConstraint
from .earth_limb import EarthLimbConstraint
from .moon_angle import MoonAngleConstraint
from .saa import SAAPolygonConstraint
from .sun_angle import SunAngleConstraint

__all__ = [
    "Constraint",
    "get_slice",
    "EarthLimbConstraint",
    "MoonAngleConstraint",
    "SunAngleConstraint",
    "SAAPolygonConstraint",
    "AltAzConstraint",
]

# Define a type that covers all constraints
Constraint = Annotated[
    EarthLimbConstraint | MoonAngleConstraint | SunAngleConstraint | SAAPolygonConstraint | AltAzConstraint,
    Field(discriminator="name"),
]
