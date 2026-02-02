from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon, points

from ...core.enums.constraint_type import ConstraintType
from ...core.schemas import AstropyDateTime
from ...ephemeris import Ephemeris
from .polygon import PolygonConstraint


class PointingConstraint(PolygonConstraint):
    """
    Polygon based pointing constraint. Defined as a Shapely Polygon with a
    start and end time. This constraint will calculate, for a given set
    of times, whether a target at a given coordinate is in the polygon.

    Attributes
    ----------
    polygon
        Shapely Polygon object defining the survey pointing.
    start_time
        AstropyDateTime defining the start of the pointing
    end_time
        AstropyDateTime defining the end of the pointing
    """

    polygon: Polygon | None = None
    name: Literal[ConstraintType.POINTING] = ConstraintType.POINTING
    short_name: str = "POINTING"
    start_time: AstropyDateTime | None = None
    end_time: AstropyDateTime | None = None

    def __call__(
        self,
        time: Time,
        coordinate: SkyCoord,
        ephemeris: Ephemeris | None = None,
    ) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate the constraint at the given time(s) and coordinate.

        Parameters
        ----------
        time : Time
            The time(s) at which to evaluate the constraint.
        coordinate : SkyCoord
            The sky coordinates to check against the constraint.
        ephemeris : Ephemeris
            The ephemeris position(s) at which to evaluate the constraint.
            This parameter is not used in this specific constraint but is included for
            compatibility with the base class.
        Returns
        -------
        ndarray
            A boolean array indicating whether the constraint is satisfied at
            the given time(s) and position(s). If time is scalar, returns a
            single boolean value. The constraint is satisfied if the
            coordinate is outside the polygon, OR if the time is outside
            the start and end time.
        """
        assert self.polygon is not None
        assert all([self.start_time is not None, self.end_time is not None])

        if all(time < self.start_time) or all(time > self.end_time):
            raise ValueError("Start and stop times outside pointing range")

        # Is the coordinate inside the pointing polygon
        # Return a boolean array of len(time)
        in_polygon = np.asarray(
            [self.polygon.contains(points(coordinate.ra.deg, coordinate.dec.deg))] * len(time)
        )

        # Are the times inside the start and end time of the pointing?
        in_pointing_time = np.array([t >= self.start_time and t < self.end_time for t in time])

        # Return the result as True or False, or an array of True/False
        # "True" means the target is constrained, because it is either outside the
        # polygon or outside the pointing time range.
        return np.asarray(np.logical_or(np.logical_not(in_polygon), np.logical_not(in_pointing_time)))
