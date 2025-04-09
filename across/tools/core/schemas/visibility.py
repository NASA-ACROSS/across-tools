from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...ephemeris.base import Ephemeris
from ..enums.constraint_type import ConstraintType
from .base import BaseSchema


class Constraint(BaseSchema, ABC):
    """
    Base class for constraints. Constraints are used to determine if a given
    coordinate is inside the constraint. This is done by checking if the
    separation between the constraint and the coordinate is less than a given
    value.

    Parameters
    ----------
    min_angle
        The minimum angle from the constraint that the spacecraft can point.
    max_angle
        The maximum angle from the constraint that the spacecraft can point.

    Methods
    -------
    __call__(time, ephemeris, coord)
        Checks if a given coordinate is inside the constraint.
    """

    short_name: ConstraintType
    name: str
    min_angle: float | None = None
    max_angle: float | None = None

    @abstractmethod
    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the constraint.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        skycoord : SkyCoord
            The coordinate to check.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")


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
