from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from ..catalogs import get_bright_stars
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
        # Get bright stars from catalog filtered by magnitude limit
        bright_stars = get_bright_stars(magnitude_limit=self.magnitude_limit)

        # Check separation from each bright star
        min_separation_rad = self.min_separation * u.deg
        in_constraint = np.zeros(len(time), dtype=bool)

        for star, _ in bright_stars:
            separation = coordinate.separation(star)
            # Note: magnitude is already filtered by get_bright_stars()
            in_constraint |= separation < min_separation_rad

        return in_constraint
