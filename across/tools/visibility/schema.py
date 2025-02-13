from ..core.enums import VisibilityType
from ..core.schemas.base import BaseSchema
from .constraints import (
    AirMassConstraint,
    AltAzConstraint,
    DayConstraint,
    EarthLimbConstraint,
    MoonConstraint,
    MoonPhaseConstraint,
    PoleConstraint,
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
    visibility_type: VisibilityType | None = None
    objvissap_url: str | None = None
    objvissap_default_params: dict | None = None
    # Loaded constraints
    constraints: list[
        EarthLimbConstraint
        | MoonConstraint
        | SAAPolygonConstraint
        | RamConstraint
        | SunConstraint
        | PoleConstraint
        | AirMassConstraint
        | MoonPhaseConstraint
        | AltAzConstraint
        | DayConstraint
        | SpaceCraftDayConstraint
    ] = []
