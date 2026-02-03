from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC, get_slice


class AirmassConstraint(ConstraintABC):
    """
    Constraint based on airmass (zenith angle) for ground-based telescopes.

    Airmass affects the quality of astronomical observations. Higher airmass
    means more atmospheric absorption and poorer image quality. This constraint
    ensures observations are conducted at acceptable airmass values.

    Parameters
    ----------
    max_airmass : float
        Maximum allowed airmass. Observations with higher airmass will be constrained.
        Typical values: 1.5-2.0 for good quality, up to 3.0 for some surveys.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the airmass is too high for the given coordinate and time.
    """

    name: Literal[ConstraintType.AIRMASS] = ConstraintType.AIRMASS
    short_name: Literal["Airmass"] = "Airmass"
    max_airmass: float = Field(default=2.0, gt=1.0, description="Maximum allowed airmass for observations")

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check if the airmass is too high for observation.

        Parameters
        ----------
        time : Time
            The time of observation.
        ephemeris : Ephemeris
            The ephemeris (used to get Earth location).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates the airmass is too high
            (constraint violated).
        """
        if ephemeris.earth_location is None:
            raise ValueError("Earth location required for airmass calculations")

        # Find the slice of ephemeris data we need
        i = get_slice(time, ephemeris)

        # Convert coordinate to AltAz for the given time and location
        altaz = coordinate.transform_to(AltAz(obstime=time[i], location=ephemeris.earth_location))

        # Calculate airmass (secant of zenith angle)
        # Airmass = sec(zenith_angle) = 1 / sin(altitude)
        airmass = 1.0 / np.sin(altaz.alt.to_value(u.rad))

        # Constrain observations with airmass above the maximum
        in_constraint: npt.NDArray[np.bool_] = airmass > self.max_airmass

        return in_constraint
