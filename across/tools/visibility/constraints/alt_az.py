from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field
from shapely import Polygon, points

from ...core.enums.constraint_type import ConstraintType
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
    min
        The minimum altitude in degrees.
    max
        The maximum altitude in degrees.
    """

    short_name: str = "AltAz"
    name: Literal[ConstraintType.ALT_AZ] = ConstraintType.ALT_AZ
    polygon: Polygon | None = None
    altitude_min: float | None = Field(default=None, ge=0, le=90)
    altitude_max: float | None = Field(default=None, ge=0, le=90)
    azimuth_min: float | None = Field(default=None, ge=0, lt=360)
    azimuth_max: float | None = Field(default=None, ge=0, lt=360)

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Calculate the Alt/Az constraint for a given time, ephemeris, and sky coordinates.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Earth location.
        coordinate : SkyCoord
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
        alt_az = coordinate.transform_to(
            AltAz(
                obstime=time[i],
                location=ephemeris.earth_location
                if ephemeris.earth_location.isscalar
                else ephemeris.earth_location[i],
            )
        )

        # Initialize the constraint array as all False
        in_constraint = np.zeros(len(alt_az), dtype=bool)

        # Calculate the basic Alt/Az min/max constraints
        if self.altitude_min is not None:
            in_constraint |= alt_az.alt < self.altitude_min * u.deg
        if self.altitude_max is not None:
            in_constraint |= alt_az.alt > self.altitude_max * u.deg
        if self.azimuth_min is not None:
            in_constraint |= alt_az.az < self.azimuth_min * u.deg
        if self.azimuth_max is not None:
            in_constraint |= alt_az.az > self.azimuth_max * u.deg

        # If a polygon is defined, then check if the Alt/Az is inside the polygon
        if self.polygon is not None:
            in_constraint |= self.polygon.contains(points(alt_az.alt, alt_az.az))

        # Return the value as a scalar or array
        return in_constraint
