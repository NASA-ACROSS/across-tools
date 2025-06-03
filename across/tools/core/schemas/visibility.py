from uuid import UUID

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
