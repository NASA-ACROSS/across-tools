from typing import Literal

import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class AirMassConstraint(Constraint):
    """
    For a given limits on airmass, is a given skycoord coordinate constrained?

    Parameters
    ----------
    airmass_min
        The minimum altitude in degrees.
    airmass_max
        The maximum altitude in degrees.
    """

    short_name: Literal["Airmass"] = "Airmass"
    name: Literal["Airmass"] = "Airmass"
    airmass_max: float | None = None
    airmass_min: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Calculate the Alt/Az constraint for a given time, ephemeris, and sky coordinates.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Earth location.
        skycoord : SkyCoord
            The sky coordinates to calculate the constraint for.

        Returns
        -------
        np.typing.NDArray[np.bool_] | np.bool_
            The calculated constraint values as a NumPy array.
        """
        # Get the range of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Convert the sky coordinates to Alt/Az coordinates
        assert ephemeris.earth_location is not None
        alt_az = skycoord.transform_to(AltAz(obstime=time[i], location=ephemeris.earth_location))

        # Initialize the constraint array as all False
        in_constraint = np.zeros(len(alt_az), dtype=bool)

        # Calculate the basic Alt/Az min/max constraints
        if self.airmass_max is not None:
            in_constraint |= alt_az.secz < self.airmass_max
        if self.airmass_min is not None:
            in_constraint |= alt_az.secz > self.airmass_min

        # Return the value as a scalar or array
        return in_constraint
