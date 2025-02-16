from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon, points

from ...ephemeris import Ephemeris
from .base import get_slice
from .polygon import PolygonConstraint


class AltAzConstraint(PolygonConstraint):
    """
    For a given Alt/Az constraint, is a given coordinate inside this
    constraint? Constraint is both defined by a polygon exclusion region and a
    minimum and maximum altitude. By default the minimum and maximum altitude
    values are 0 and 90 degrees respectively. Polygon restriction regions can
    be combined with minimum and maximum altitude restrictions.

    Parameters
    ----------
    polygon
        The polygon defining the exclusion region.
    alt_min
        The minimum altitude in degrees.
    alt_max
        The maximum altitude in degrees.
    """

    short_name: Literal["Alt/Az"] = "Alt/Az"
    name: Literal["Altitude/Azimuth Avoidance"] = "Altitude/Azimuth Avoidance"
    polygon: Polygon | None
    alt_min: float | None
    alt_max: float | None

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
        np.ndarray
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
        if self.alt_min is not None:
            in_constraint |= alt_az.alt < self.alt_min * u.deg
        if self.alt_max is not None:
            in_constraint |= alt_az.alt > self.alt_max * u.deg

        # If a polygon is defined, then check if the Alt/Az is inside the polygon
        if self.polygon is not None:
            in_constraint |= self.polygon.contains(points(alt_az.alt, alt_az.az))

        # Return the value as a scalar or array
        return in_constraint
