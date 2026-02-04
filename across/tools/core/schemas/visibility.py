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
    A class to hold computed values for used by in constraint calculations.
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
    air_mass: u.Quantity | None = Field(default=None, description="Airmass value for the coordinate")
    sun_altitude: u.Quantity | None = Field(default=None, description="Altitude of the Sun at the given time")

    def merge(self, other: "VisibilityComputedValues") -> None:
        """
        Merge another VisibilityComputedValues into this one. Always updates
        values from the other object if they are not None.
        """
        for field_name, value in other.model_dump().items():
            if value is not None:
                setattr(self, field_name, value)
