from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.schemas.visibility import ConstraintType
from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class SpaceCraftDayConstraint(Constraint):
    """
    A constraint that determines if a spacecraft is in daylight. This
    constraint checks whether the spacecraft (observatory) is in daytime,
    defined as when any part of the Sun is above the Earth's limb from the
    spacecraft's perspective.

    Parameters
    ----------
    whole_sun : bool, optional
        If True, considers the entire solar disk when calculating the
        constraint. If False, only considers the center of the Sun, so
        nighttime begins when the Sun is >50% below the horizon. Default is
        True.

    Attributes
    ----------
    name : Literal["Spacecraft Daytime"]
        Full name of the constraint
    short_name : Literal["Day"]
        Abbreviated name of the constraint

    Methods
    -------
    __call__(time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]
        Evaluates the daytime constraint at given time(s)

    Notes
    -----
    The constraint uses ephemeris data to calculate the angular separation
    between the Earth and Sun as seen from the spacecraft's location. It
    accounts for the apparent sizes of both the Earth and Sun when determining
    if the spacecraft is in daylight.
    """

    name: Literal["Spacecraft Daytime"] = "Spacecraft Daytime"
    short_name: Literal[ConstraintType.ORBIT_DAY] = ConstraintType.ORBIT_DAY
    whole_sun: bool = True

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        For a given time, ephemeris, check if the observatory is in daytime.
        Daytime is defined as the time when any part of the Sun is above the
        Earth Limb.

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
        assert (
            ephemeris.sun is not None
            and ephemeris.earth is not None
            and ephemeris.earth_radius_angle is not None
            and ephemeris.sun_radius_angle is not None
            and ephemeris.earth_location is not None
        )
        # Calculate the separation between the Earth and the Sun
        sun_earth_sep = ephemeris.sun[i].separation(ephemeris.earth[i])
        if self.whole_sun:
            in_constraint = np.array(
                sun_earth_sep > (ephemeris.earth_radius_angle - ephemeris.sun_radius_angle)
            )
        else:
            in_constraint = sun_earth_sep > ephemeris.earth_radius_angle

        # Return the
        return in_constraint[0] if time.isscalar else in_constraint
