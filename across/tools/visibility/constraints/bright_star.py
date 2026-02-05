from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC


class BrightStarConstraint(ConstraintABC):
    """
    Constraint that avoids observing too close to bright stars.

    Bright stars can cause detector saturation, blooming, or scattered light
    issues. This constraint ensures observations are conducted at sufficient
    angular distance from bright stars.

    Parameters
    ----------
    min_separation : float
        Minimum angular separation (degrees) required from bright stars.
        Observations closer than this to bright stars will be constrained.
    magnitude_limit : float
        Magnitude limit for bright stars to avoid. Stars brighter than this
        magnitude will be considered for avoidance. Default is 6.0 (naked eye visible).

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate is too close to any bright star.
    """

    name: Literal[ConstraintType.BRIGHT_STAR] = ConstraintType.BRIGHT_STAR
    short_name: Literal["Bright Star"] = "Bright Star"
    min_separation: float = Field(
        default=5.0, gt=0, description="Minimum angular separation (degrees) from bright stars"
    )
    magnitude_limit: float = Field(
        default=6.0, description="Magnitude limit for stars to avoid (brighter than this)"
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check if the coordinate is too close to any bright star.

        Parameters
        ----------
        time : Time
            The time (not used for star positions).
        ephemeris : Ephemeris
            The ephemeris (not used for star positions).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to a bright star).
        """
        # For now, implement a simplified version with major bright stars
        # In a real implementation, this would use a star catalog
        # Here we include some of the brightest stars as examples

        bright_stars = [
            SkyCoord(ra="05h16m41.4s", dec="-08d12m05.9s"),  # Sirius
            SkyCoord(ra="06h45m08.9s", dec="-16d42m58.0s"),  # Canopus
            SkyCoord(ra="07h39m18.1s", dec="05d13m30.0s"),  # Arcturus
            SkyCoord(ra="14h39m36.5s", dec="-60d50m02.3s"),  # Alpha Centauri
            SkyCoord(ra="18h36m56.3s", dec="38d47m01.3s"),  # Vega
            SkyCoord(ra="20h41m25.9s", dec="45d16m49.3s"),  # Capella
            SkyCoord(ra="22h57m39.0s", dec="-29d37m20.0s"),  # Rigel
            SkyCoord(ra="05h14m32.3s", dec="-08d12m05.9s"),  # Procyon
            SkyCoord(ra="01h09m43.9s", dec="35d37m14.0s"),  # Betelgeuse
            SkyCoord(ra="17h45m40.0s", dec="-29d00m28.0s"),  # Achernar
        ]

        # Check separation from each bright star
        min_separation_rad = self.min_separation * u.deg
        in_constraint = np.zeros(len(time), dtype=bool)

        for star in bright_stars:
            separation = coordinate.separation(star)
            in_constraint |= separation < min_separation_rad

        return in_constraint
