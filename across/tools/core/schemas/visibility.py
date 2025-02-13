from datetime import datetime
from typing import Optional
from .base import BaseSchema


class VisWindow(BaseSchema):
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

    begin: datetime
    end: datetime
    visibility: int
    initial: str
    final: str


class ConstrainedDate(BaseSchema):
    datetime: datetime
    constraint: Constraint
    observatory_id: int


class Window(BaseSchema):
    begin: ConstrainedDate
    end: ConstrainedDate


class VisibilityWindow(BaseSchema):
    window: Window
    max_visibility_duration: int


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
    min_vis: Optional[int] = None
    hires: bool = True
    entries: list[VisWindow] = []
