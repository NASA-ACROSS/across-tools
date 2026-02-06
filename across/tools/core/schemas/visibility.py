from uuid import UUID

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from pydantic import Field

from ..enums.constraint_type import ConstraintType
from .base import BaseSchema
from .custom_types import AstropyDateTime


class ConstrainedDate(BaseSchema):
    """
    Represents a constrained date with associated constraint information.

    Attributes
    ----------
    datetime
        The time of the constrained date.
    constraint
        The type of constraint active at this time.
    observatory_id
        UUID of the observatory or instrument associated with this constraint.

    """

    datetime: AstropyDateTime
    constraint: ConstraintType
    observatory_id: UUID


class Window(BaseSchema):
    """
    Represents a time window bounded by constrained dates.

    Attributes
    ----------
    begin
        The start of the window with its associated constraint.
    end
        The end of the window with its associated constraint.

    """

    begin: ConstrainedDate
    end: ConstrainedDate


class ConstraintReason(BaseSchema):
    """
    Represents the reasons for constraints at the boundaries of a visibility window.

    Attributes
    ----------
    start_reason
        Human-readable description of the constraint that ends before the window starts.
    end_reason
        Human-readable description of the constraint that begins after the window ends.

    """

    start_reason: str
    end_reason: str


class VisibilityWindow(BaseSchema):
    """
    Represents a visibility window with duration and constraint information.

    Attributes
    ----------
    window
        The time window defining the start and end of the visibility period.
    max_visibility_duration
        Maximum visibility duration in seconds for this window.
    constraint_reason
        The reasons for constraints at the window boundaries.

    """

    window: Window
    max_visibility_duration: int
    constraint_reason: ConstraintReason


class VisibilityComputedValues(BaseSchema):
    """
    A class to hold computed values used in constraint calculations.

    Attributes
    ----------
    sun_angle
        Angular distance between the Sun and the target coordinate.
    moon_angle
        Angular distance between the Moon and the target coordinate.
    earth_angle
        Angular distance between the Earth limb and the target coordinate.
    alt_az
        Altitude-azimuth coordinates of the target from the observatory.

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

    def merge(self, other: "VisibilityComputedValues") -> None:
        """
        Merge another VisibilityComputedValues into this one.

        Always updates values from the other object if they are not None.

        Parameters
        ----------
        other
            Another VisibilityComputedValues instance to merge from.

        """
        for field_name, value in other.model_dump().items():
            if value is not None:
                setattr(self, field_name, value)
