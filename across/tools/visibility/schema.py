from typing import Any

from ..core.schemas.base import BaseSchema
from ..core.schemas.coordinate import DateRangeSchema
from .constraints import (
    AirMassConstraint,
    AltAzConstraint,
    DayConstraint,
    EarthLimbConstraint,
    MoonConstraint,
    MoonPhaseConstraint,
    OrbitPoleConstraint,
    RamConstraint,
    SAAPolygonConstraint,
    SpaceCraftDayConstraint,
    SunConstraint,
)


class ObservatoryConstraints(BaseSchema):
    """
    Class to define the constraint configuration for calculating target
    visibililty for a given observatory.
    """

    # observatory_id: int
    # id: int | None = None
    # created_by: str | None = Field(None, exclude=True)
    # created_on: datetime | None = None
    # modified_on: datetime | None = None
    # modified_by: str | None = Field(None, exclude=True)
    objvissap_url: str | None = None
    objvissap_default_params: dict[str, Any] | None = None
    # Loaded constraints
    constraints: list[
        EarthLimbConstraint
        | MoonConstraint
        | SAAPolygonConstraint
        | RamConstraint
        | SunConstraint
        | OrbitPoleConstraint
        | AirMassConstraint
        | MoonPhaseConstraint
        | AltAzConstraint
        | DayConstraint
        | SpaceCraftDayConstraint
    ] = []


class VisWindow(DateRangeSchema):
    """
    Represents a visibility window.

    Parameters
    ----------
    begin
        The beginning of the window.
    end
        The end of the window.
    initial
        The main constraint that ends at the beginning of the window.
    final
        The main constraint that begins at the end of the window.
    visibility
        The amount of seconds in the window that the object is visible.
    """

    visibility: int
    initial: str
    final: str
