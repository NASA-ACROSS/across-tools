from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC, get_slice


class SunAngleConstraint(ConstraintABC):
    """
    For a given Sun avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Sun that the spacecraft can point.

    max_angle
        The maximum angle from the Sun that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, sun_radius_angle=None)
        Checks if a given coordinate is inside the constraint.

    """

    name: Literal[ConstraintType.SUN] = ConstraintType.SUN
    short_name: Literal["Sun"] = "Sun"
    min_angle: float | None = Field(default=None, ge=0, le=180, description="Minimum angle from the Sun")
    max_angle: float | None = Field(default=None, ge=0, le=180, description="Maximum angle from the Sun")

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Sun constraint. This is done by checking if the
        separation between the Sun and the spacecraft is less than the
        a given minimum angle or greater than a maximum angle (essentially an
        anti-Sun constraint).

        Parameters
        ----------
        coordinate : SkyCoord
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

        # Calculate the angular distance between the center of the Sun and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `sun` is 3x faster than just doing the separation
        # directly with `sun`.
        assert ephemeris.sun is not None

        # Construct the constraint based on the minimum and maximum angles
        in_constraint = np.zeros(len(ephemeris.sun[i]), dtype=bool)

        if self.min_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.sun[i].ra, ephemeris.sun[i].dec).separation(coordinate)
                < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.sun[i].ra, ephemeris.sun[i].dec).separation(coordinate)
                > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint
