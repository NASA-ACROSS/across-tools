from typing import Annotated

from pydantic import Field

from .alt_az import AltAzConstraint
from .base import get_slice
from .earth_limb import EarthLimbConstraint
from .logical import AndConstraint, NotConstraint, OrConstraint, XorConstraint
from .moon_angle import MoonAngleConstraint
from .saa import SAAPolygonConstraint
from .sun_angle import SunAngleConstraint

__all__ = [
    "AllConstraint",
    "get_slice",
    "EarthLimbConstraint",
    "MoonAngleConstraint",
    "SunAngleConstraint",
    "SAAPolygonConstraint",
    "AltAzConstraint",
    "AndConstraint",
    "OrConstraint",
    "NotConstraint",
    "XorConstraint",
]

# Define a type that covers all constraints
AllConstraint = Annotated[
    EarthLimbConstraint
    | MoonAngleConstraint
    | SunAngleConstraint
    | SAAPolygonConstraint
    | AltAzConstraint
    | AndConstraint
    | OrConstraint
    | NotConstraint
    | XorConstraint,
    Field(discriminator="name"),
]

# Rebuild models to resolve forward references
AndConstraint.model_rebuild()
OrConstraint.model_rebuild()
NotConstraint.model_rebuild()
XorConstraint.model_rebuild()
