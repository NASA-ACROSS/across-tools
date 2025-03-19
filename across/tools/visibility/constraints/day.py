from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.enums import TwilightType
from ...core.schemas.visibility import ConstraintType
from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class DayConstraint(Constraint):
    """
    For a given daytime constraint, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    twilight_type : TwilightType
        The type of twilight to consider bounding "daytime". Options are
        "astronomical", "nautical", "civil", and "sunrise".
    horizon_dip : bool, optional
        If True, consider the dip of the horizon due to the elevation of
        the observatory. Default is False.

    """

    short_name: Literal[ConstraintType.DAY] = ConstraintType.DAY
    name: Literal["Daytime"] = "Daytime"
    twilight_type: TwilightType
    horizon_dip: bool = False

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        For a given time, ephemeris, check if it is daytime at the observatory.
        Daytime is defined by the twilight type, and whether the Sun is above
        the horizon.

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
        np.ndarray
            Boolean array of `True` if the coordinate is inside the constraint, `False`
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
            and ephemeris.timestamp is not None
            and ephemeris.earth_location is not None
            and ephemeris.sun_radius_angle is not None
        )
        sun_alt = (
            ephemeris.sun[i]
            .transform_to(AltAz(obstime=ephemeris.timestamp[i], location=ephemeris.earth_location))
            .alt
        )

        # Calculate the horizon dip due to elevation of the observatory
        if self.horizon_dip:
            height = ephemeris.earth_location.height.to_value(u.m)
            dip_horizon_degrees = np.sqrt(2 * height) * 0.034 * u.deg
        else:
            dip_horizon_degrees = 0 * u.deg

        # Depending on the twilight type, calculate if it is considered day time
        if self.twilight_type == TwilightType.ASTRONOMICAL:
            in_constraint = np.array(sun_alt > (-18 * u.deg - dip_horizon_degrees))
        elif self.twilight_type == TwilightType.NAUTICAL:
            in_constraint = np.array(sun_alt > (-12 * u.deg - dip_horizon_degrees))
        elif self.twilight_type == TwilightType.CIVIL:
            in_constraint = np.array(sun_alt > (-6 * u.deg - dip_horizon_degrees))
        elif self.twilight_type == TwilightType.SUNRISE:
            in_constraint = np.array(sun_alt > 0 * u.deg)

        # Return the
        return in_constraint[0] if time.isscalar else in_constraint
