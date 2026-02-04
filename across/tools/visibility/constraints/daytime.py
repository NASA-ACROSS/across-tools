from enum import Enum
from typing import Literal

import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris, GroundEphemeris
from .base import ConstraintABC, get_slice


class TwilightType(str, Enum):
    """Enumeration of twilight types for daytime constraints."""

    ASTRONOMICAL = "astronomical"  # Sun 12-18° below horizon
    NAUTICAL = "nautical"  # Sun 6-12° below horizon
    CIVIL = "civil"  # Sun 0-6° below horizon
    SUNSET = "sunset"  # Sun at horizon


class DaytimeConstraint(ConstraintABC):
    """
    Constraint that avoids daytime observations based on twilight definitions.

    For ground-based telescopes, daytime is defined by the Sun's position relative
    to the horizon. Different twilight types provide different levels of darkness.

    For space-based telescopes, daytime refers to periods when the spacecraft
    is in direct sunlight (requires appropriate ephemeris data).

    The telescope type (ground vs space) is automatically detected based on the
    ephemeris type: GroundEphemeris for ground-based, all others for space-based.

    Parameters
    ----------
    twilight_type : TwilightType
        The type of twilight to use for defining daytime boundaries.
        - 'astronomical': Sun 12-18° below horizon (default)
        - 'nautical': Sun 6-12° below horizon
        - 'civil': Sun 0-6° below horizon
        - 'day': Sun above horizon

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the observation time qualifies as daytime for the given constraints.
    """

    name: Literal[ConstraintType.DAYTIME] = ConstraintType.DAYTIME
    short_name: Literal["Daytime"] = "Daytime"
    twilight_type: TwilightType = Field(
        default=TwilightType.ASTRONOMICAL, description="Type of twilight defining daytime boundaries"
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check if the observation time is during daytime.

        Parameters
        ----------
        time : Time
            The time of observation.
        ephemeris : Ephemeris
            The ephemeris (used for Earth location and potentially sunlight data).
        coordinate : SkyCoord
            The coordinate (not used for daytime calculations).

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates daytime (constraint violated).
        """
        # Auto-detect telescope type based on ephemeris type
        if isinstance(ephemeris, GroundEphemeris):
            return self._check_ground_daytime(time, ephemeris)
        else:
            # Assume space-based for all other ephemeris types
            return self._check_space_daytime(time, ephemeris)

    def _check_ground_daytime(self, time: Time, ephemeris: Ephemeris) -> np.typing.NDArray[np.bool_]:
        """
        Check daytime for ground-based telescopes using Sun altitude.
        """
        # Find the slice of ephemeris data we need
        i = get_slice(time, ephemeris)

        # Get Sun position
        sun_altaz = ephemeris.sun[i].transform_to(AltAz(obstime=time[i], location=ephemeris.earth_location))

        self.computed_values.sun_altitude = sun_altaz.alt

        # Define daytime based on twilight type
        if self.twilight_type == TwilightType.SUNSET:
            # Twilight starts after sunset
            in_constraint = self.computed_values.sun_altitude.deg > -0.866
        elif self.twilight_type == TwilightType.CIVIL:
            # Sun within 6° of horizon (civil twilight)
            in_constraint = self.computed_values.sun_altitude.deg > -6
        elif self.twilight_type == TwilightType.NAUTICAL:
            # Sun within 12° of horizon (nautical twilight)
            in_constraint = self.computed_values.sun_altitude.deg > -12
        elif self.twilight_type == TwilightType.ASTRONOMICAL:
            # Sun within 18° of horizon (astronomical twilight)
            in_constraint = self.computed_values.sun_altitude.deg > -18

        return np.asarray(in_constraint, dtype=bool)

    def _check_space_daytime(self, time: Time, ephemeris: Ephemeris) -> np.typing.NDArray[np.bool_]:
        """
        Check daytime for space-based telescopes using proper eclipse calculations.

        Determines if the spacecraft is in direct sunlight by calculating whether
        the Earth is blocking the line of sight to the Sun, accounting for the
        finite sizes of both Earth and Sun.
        """
        # Find the slice of ephemeris data we need
        i = get_slice(time, ephemeris)

        # Calculate angular separation between Sun and Earth centers as seen from spacecraft
        sun_earth_separation = ephemeris.sun[i].separation(ephemeris.earth[i])

        # Spacecraft is in eclipse (not in sunlight) if the angular separation
        # between Sun and Earth centers is less than the sum of their angular radii
        # This means the Earth is blocking the Sun from the spacecraft's perspective
        in_eclipse = sun_earth_separation < ephemeris.earth_radius_angle[i] + ephemeris.sun_radius_angle[i]

        # For space telescopes, daytime means the spacecraft is in sunlight
        # So in_constraint is True when NOT in eclipse (i.e., in sunlight)
        in_constraint = ~in_eclipse

        return np.asarray(in_constraint, dtype=bool)
