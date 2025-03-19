from datetime import datetime
from enum import Enum

from across.tools.visibility.constraints.base import Constraint

from .base import BaseSchema


class Constraints(BaseSchema):
    """
    Represents a list of constraints.

    Parameters
    ----------
    constraints
        List of constraints.
    """

    constraint_type: str
    constraints: list[Constraint]


class ConstraintType(str, Enum):
    """
    Represents a constraint.
    """

    SUN = "Sun"
    MOON = "Moon"
    EARTH = "Earth"
    UNKNOWN = "Unknown"
    WINDOW = "Window"
    VISIBILITY = "Visibility"
    DAY = "Day"
    MOON_PHASE = "Moon Phase"
    ORBIT_DAY = "Orbit Day"
    ORBIT_POLE = "Orbit Pole"
    ORBIT_RAM = "Orbit Ram"
    ORBIT_SAA = "SAA"
    AIR_MASS = "Air Mass"
    ALT_AZ = "Alt/Az"


class ConstrainedDate(BaseSchema):
    """
    Represents a constrained date.
    """

    datetime: datetime
    constraint: ConstraintType
    observatory_id: str


class Window(BaseSchema):
    """Visibility Window"""

    begin: ConstrainedDate
    end: ConstrainedDate


class VisibilityWindow(BaseSchema):
    """Visibility Window"""

    window: Window
    max_visibility_duration: int
    constraint_reason: "ConstraintReason"


class ConstraintReason(BaseSchema):
    """
    Represents the reasons for constraints.
    """

    start_reason: str
    end_reason: str


class VisibilitySchema(BaseSchema):
    """
    Schema for visibility classes.

    Parameters
    ----------
    entries: List[VisWindow]
        List of visibility windows.

        Information about the job status.
    """

    observatory_id: int
    min_vis: int | None = None
    hires: bool = True
    entries: list[VisibilityWindow] = []
