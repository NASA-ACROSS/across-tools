from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC, get_slice


class MoonAngleConstraint(ConstraintABC):
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

    name: Literal[ConstraintType.MOON] = ConstraintType.MOON
    short_name: Literal["Moon"] = "Moon"
    min_angle: float | None = Field(default=None, ge=0, le=180, description="Minimum angle from the Moon")
    max_angle: float | None = Field(default=None, ge=0, le=180, description="Maximum angle from the Moon")

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Moon constraint. This is done by checking if the
        separation between the Moon and the spacecraft is less than the
        Moon's angular radius plus the minimum angle.

        Parameters
        ----------
        coordinate : SkyCoord:
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

        in_constraint = np.zeros(len(ephemeris.moon[i]), dtype=bool)
        if self.min_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(coordinate)
                < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(coordinate)
                > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint
