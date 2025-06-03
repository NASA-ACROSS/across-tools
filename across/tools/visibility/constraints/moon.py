from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.enums.constraint_type import ConstraintType
from ...core.schemas.visibility import Constraint
from ...ephemeris import Ephemeris
from .base import get_slice


class MoonConstraint(Constraint):
    """
    For a given Moon avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Moon that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, moon_radius_angle=None)
        Checks if a given coordinate is inside the constraint.

    """

    short_name: Literal[ConstraintType.MOON] = ConstraintType.MOON
    name: Literal["Moon Angle"] = "Moon Angle"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Moon constraint. This is done by checking if the
        separation between the Moon and the spacecraft is less than the
        Moon's angular radius plus the minimum angle.

        Parameters
        ----------
        skycoord : SkyCoord:
            The coordinate to check.
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Moon and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `moon` is 3x faster than just doing the separation
        # directly with `moon`.
        assert ephemeris.moon is not None

        # FIXME: Add option to support size of moon in angle

        in_constraint = np.zeros(len(ephemeris.moon[i]), dtype=bool)
        if self.min_angle is None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(skycoord)
                < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(skycoord)
                > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint
