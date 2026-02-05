from typing import Annotated

from pydantic import Field

from .airmass import AirmassConstraint
from .alt_az import AltAzConstraint
from .base import get_slice
from .bright_star import BrightStarConstraint
from .daytime import DaytimeConstraint
from .earth_limb import EarthLimbConstraint
from .ecliptic_latitude import EclipticLatitudeConstraint
from .galactic_bulge import GalacticBulgeConstraint
from .galactic_plane import GalacticPlaneConstraint
from .logical import AndConstraint, NotConstraint, OrConstraint, XorConstraint
from .moon_angle import MoonAngleConstraint
from .saa import SAAPolygonConstraint
from .solar_system import SolarSystemConstraint
from .sun_angle import SunAngleConstraint

__all__ = [
    "AllConstraint",
    "get_slice",
    "EarthLimbConstraint",
    "MoonAngleConstraint",
    "SunAngleConstraint",
    "SAAPolygonConstraint",
    "AltAzConstraint",
    "GalacticPlaneConstraint",
    "BrightStarConstraint",
    "AirmassConstraint",
    "EclipticLatitudeConstraint",
    "GalacticBulgeConstraint",
    "SolarSystemConstraint",
    "DaytimeConstraint",
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
    | GalacticPlaneConstraint
    | BrightStarConstraint
    | AirmassConstraint
    | EclipticLatitudeConstraint
    | GalacticBulgeConstraint
    | SolarSystemConstraint
    | DaytimeConstraint
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
