from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon, points

from ...core.enums.constraint_type import ConstraintType
from ...ephemeris import Ephemeris
from .base import get_slice
from .polygon import PolygonConstraint


class SAAPolygonConstraint(PolygonConstraint):
    """
    Polygon based SAA constraint. The SAA is defined by a Shapely Polygon, and
    this constraint will calculate for a given set of times and a given
    ephemeris whether the spacecraft is in that SAA polygon.

    Attributes
    ----------
    polygon
        Shapely Polygon object defining the SAA polygon.
    """

    polygon: Polygon | None = None
    name: Literal[ConstraintType.SAA] = ConstraintType.SAA
    short_name: str = "SAA"

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Evaluate the constraint at the given time(s) and ephemeris position(s).

        Parameters
        ----------
        time : Time
            The time(s) at which to evaluate the constraint.
        ephemeris : Ephemeris
            The ephemeris position(s) at which to evaluate the constraint.
        coordinate : SkyCoord
            The sky coordinates to check against the constraint. This parameter
            is not used in this specific constraint but is included for
            compatibility with the base class.

        Returns
        -------
        ndarray
            A boolean array indicating whether the constraint is satisfied at
            the given time(s) and position(s). If time is scalar, returns a
            single boolean value.
        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)
        if ephemeris.longitude is None or ephemeris.latitude is None:
            raise ValueError("Ephemeris must contain longitude and latitude")
        assert self.polygon is not None
        in_constraint = np.array(
            self.polygon.contains(
                points(ephemeris.longitude[i].to_value(u.deg), ephemeris.latitude[i].to_value(u.deg))
            )
        )

        # Return the result as True or False, or an array of True/False
        return in_constraint
