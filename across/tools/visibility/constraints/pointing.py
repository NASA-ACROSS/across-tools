from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.enums.constraint_type import ConstraintType
from ...core.schemas import Coordinate
from ...ephemeris import Ephemeris
from ...footprint import Pointing
from .base import ConstraintABC


class PointingConstraint(ConstraintABC):
    """
    Footprint based pointing constraint. A Pointing is defined
    as a Footprint with a start and end time. This constraint will
    calculate whether a target was within any set of pointings.

    Attributes
    ----------
    pointings
        List of Pointing objects.
    start_time
        AstropyDateTime defining the start of the pointing
    end_time
        AstropyDateTime defining the end of the pointing
    """

    pointings: list[Pointing]
    name: Literal[ConstraintType.POINTING] = ConstraintType.POINTING
    short_name: str = "POINTING"

    def __call__(
        self,
        time: Time,
        ephemeris: Ephemeris,
        coordinate: SkyCoord,
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
            coordinate is outside the footprint, OR if the time is outside
            the start and end time.
        """
        # For each pointing, is the coordinate within the footprint
        # and time range?
        # Return a boolean array of len(time)
        in_constraint = np.ones(len(time), dtype=bool)

        for pointing in self.pointings:
            # Is the target inside the footprint of the pointing?
            in_footprint = np.asarray(
                [
                    pointing.footprint.contains(Coordinate(ra=coordinate.ra.deg, dec=coordinate.dec.deg))
                    * len(time)
                ],
                dtype=bool,
            )

            # Are the times inside the start and end time of the pointing?
            in_pointing_time = np.array([t >= pointing.start_time and t < pointing.end_time for t in time])

            # Return the result as True or False, or an array of True/False
            # "True" means the target is constrained, because it is either outside the
            # footprint or outside the pointing time range.
            in_constraint &= np.logical_or(np.logical_not(in_footprint), np.logical_not(in_pointing_time))

        return in_constraint
