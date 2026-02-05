from typing import Literal

import numpy as np
import numpy.typing as npt
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC


class EclipticLatitudeConstraint(ConstraintABC):
    """
    Constraint that avoids observing at low ecliptic latitudes.

    The ecliptic plane contains zodiacal dust that scatters sunlight, creating
    zodiacal light that can contaminate observations. This constraint ensures
    observations are conducted at sufficient ecliptic latitude to avoid this
    background light.

    Parameters
    ----------
    min_latitude : float
        Minimum ecliptic latitude (degrees) required for observation.
        Observations closer to the ecliptic plane than this will be constrained.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate is too close to the ecliptic plane.
    """

    name: Literal[ConstraintType.ECLIPTIC_LATITUDE] = ConstraintType.ECLIPTIC_LATITUDE
    short_name: Literal["Ecliptic Latitude"] = "Ecliptic Latitude"
    min_latitude: float = Field(
        default=15.0, ge=0, le=90, description="Minimum ecliptic latitude (degrees) for valid observations"
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check if the coordinate is too close to the ecliptic plane.

        Parameters
        ----------
        time : Time
            The time (not used for ecliptic coordinates).
        ephemeris : Ephemeris
            The ephemeris (not used for ecliptic coordinates).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to the ecliptic plane).
        """
        # Convert to ecliptic coordinates
        ecliptic_coord = coordinate.transform_to(GeocentricTrueEcliptic())

        # Check if the absolute ecliptic latitude is less than the minimum
        # This creates a "zone of avoidance" around the ecliptic plane
        in_constraint: npt.NDArray[np.bool_] = np.abs(ecliptic_coord.lat.deg) < self.min_latitude

        return in_constraint
