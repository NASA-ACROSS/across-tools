from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.schemas.visibility import ConstraintType
from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class RamConstraint(Constraint):
    """
    For a given Ram avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Ram that the spacecraft can point.
    max_angle
        The maximum angle from the Ram that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, ramsize=None)
        Checks if a given coordinate is inside the constraint.

    """

    name: Literal["Ram Direction Constraint"] = "Ram Direction Constraint"
    short_name: Literal[ConstraintType.ORBIT_RAM] = ConstraintType.ORBIT_RAM
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Ram direction constraint. Ram direction is the the direction
        the spacecraft is travelling in, and is often avoided due to an
        increased chance of particles entering the telescope if pointed along
        the direction of motion. This is done by checking if the separation
        between the spacecraft velocity vector and the spacecraft pointing is
        less than the minimum angle allowed or the maximum allowed angle.

        Parameters
        ----------
        skycoord
            The coordinate to check.
        time
            The time to check.
        ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Ram
        # direction and the object. Note that creating the SkyCoord here from
        # ra/dec stored in the ephemeris `ram` is 3x faster than just doing the
        # separation directly with `ram`.
        velvec = ephemeris.gcrs.velocity.d_xyz.value
        assert velvec is not None

        in_constraint = np.zeros(len(velvec[i]), dtype=bool)

        if self.min_angle is None:
            in_constraint |= SkyCoord(velvec[i]).separation(skycoord) < self.min_angle * u.deg
        if self.max_angle is not None:
            in_constraint |= SkyCoord(velvec[i]).separation(skycoord) > self.max_angle * u.deg

        # Return the result as True or False, or an array of True/False
        return in_constraint
