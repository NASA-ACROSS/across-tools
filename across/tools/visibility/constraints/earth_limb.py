from typing import Literal

import astropy.units as u  # type: ignore[import]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import]
from astropy.time import Time  # type: ignore[import]

from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class EarthLimbConstraint(Constraint):
    """
    For a given Earth limb avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    name
        The name of the constraint.
    short_name
        The short name of the constraint.
    min_angle
        The minimum angle from the Earth limb that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, earth_radius_angle=None)
        Checks if a given coordinate is inside the constraint.
    """

    short_name: Literal["Earth"] = "Earth"
    name: Literal["Earth Limb Constraint"] = "Earth Limb Constraint"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Earth limb constraint. This is done by checking if the
        separation between the Earth and the spacecraft is less than the
        Earth's angular radius plus the minimum angle.

        NOTE: Assumes a circular approximation for Earth.

        Parameters
        ----------
        skycoord : SkyCoord
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

        # Calculate the angular distance between the center of the Earth and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `earth` is 3x faster than just doing the separation
        # directly with `earth`.
        assert ephemeris.earth is not None and ephemeris.earth_radius_angle is not None

        in_constraint = np.zeros(len(ephemeris.earth[i]), dtype=bool)

        if self.min_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.earth[i].ra, ephemeris.earth[i].dec).separation(skycoord)
                < ephemeris.earth_radius_angle[i] + self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.earth[i].ra, ephemeris.earth[i].dec).separation(skycoord)
                > ephemeris.earth_radius_angle[i] + self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint
