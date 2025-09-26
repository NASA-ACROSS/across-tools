from uuid import UUID

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from pydantic import Field

from ..enums.constraint_type import ConstraintType
from .base import BaseSchema
from .custom_types import AstropyDateTime


class ConstrainedDate(BaseSchema):
    """
    Represents a constrained date.
    """

    datetime: AstropyDateTime
    constraint: ConstraintType
    observatory_id: UUID


class Window(BaseSchema):
    """Visibility Window"""

    begin: ConstrainedDate
    end: ConstrainedDate


class ConstraintReason(BaseSchema):
    """
    Represents the reasons for constraints.
    """

    start_reason: str
    end_reason: str


class VisibilityWindow(BaseSchema):
    """Visibility Window"""

    window: Window
    max_visibility_duration: int
    constraint_reason: ConstraintReason


class VisibilityComputedValues(BaseSchema):
    """
    A class to hold computed values for the SunAngleConstraint.
    """

    sun_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Sun and the coordinate"
    )
    moon_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Moon and the coordinate"
    )
    earth_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Earth and the coordinate"
    )
    alt_az: SkyCoord | None = Field(default=None, description="AltAz coordinates of the coordinate")
