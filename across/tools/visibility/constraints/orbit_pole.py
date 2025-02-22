from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class OrbitPoleConstraint(Constraint):
    """
    For a given pole avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the pole that the spacecraft can point.
    max_angle
        The maximum angle from the pole that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, polesize=None)
        Checks if a given coordinate is inside the constraint.

    """

    short_name: Literal["Pole"] = "Pole"
    name: Literal["Orbit Pole Angle"] = "Orbit Pole Angle"
    min_angle: float | None = None
    max_angle: float | None = None
    earth_limb_pole: bool = True

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Pole constraint. The pole constraint is defined as the
        minimum angle away from the poles of the orbit that the spacecraft can
        be pointed.

        This is done by checking if the separation between the Pole and the
        spacecraft is less than the Pole's angular radius plus the minimum
        angle.

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

        # Calculate the angular distance between the center of the Pole and the
        # object. Note that creating the SkyCoord here from ra/dec stored in
        # the ephemeris `pole` is 3x faster than just doing the separation
        # directly with `pole`.

        # Find the vector of the orbit pole, which is cross product of the
        # orbit vector and the velocity vector

        # Calculate the pole vector
        pole = SkyCoord(ephemeris.gcrs.cartesian.without_differentials().cross(ephemeris.gcrs.velocity))

        # If this is an Earth Limb driven pole constraint, then we need to
        # calculate the pole from the Earth limb constraint, in this case
        # min_angle is the minimum angle from the Earth limb that the
        # spacecraft can point.
        if self.earth_limb_pole and self.min_angle is not None:
            min_angle = ephemeris.earth_radius_angle[i].to_value("deg") + (self.min_angle - 90)
        else:
            # Assume self.min_angle is minimum angle we can be from the pole
            min_angle = self.min_angle

        # Calculate constraint as being within the min_angle of the northern or
        # southern pole
        polesep = pole.separation(skycoord)

        # If the pole separation is greater than 90 degrees, then we're
        # actually closer to the other pole
        index = np.where(polesep.value > 90)
        polesep[index] = 180 * u.deg - polesep[index]

        in_constraint = np.zeros(len(polesep), dtype=bool)
        if self.min_angle is not None:
            in_constraint |= polesep.value < min_angle
        if self.max_angle is not None:
            in_constraint |= polesep.value > self.max_angle

        # Return the result as True or False, or an array of True/False
        return in_constraint
