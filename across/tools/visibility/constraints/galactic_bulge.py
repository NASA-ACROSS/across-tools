from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC


class GalacticBulgeConstraint(ConstraintABC):
    """
    Constraint that avoids observing too close to the Galactic Bulge.

    The Galactic Bulge is a crowded region with extremely high stellar density,
    making it difficult to observe individual sources. This constraint ensures
    observations avoid this region.

    Parameters
    ----------
    min_separation : float
        Minimum angular separation (degrees) required from the Galactic Bulge.
        Observations closer than this will be constrained.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate is too close to the Galactic Bulge.
    """

    name: Literal[ConstraintType.GALACTIC_BULGE] = ConstraintType.GALACTIC_BULGE
    short_name: Literal["Galactic Bulge"] = "Galactic Bulge"
    min_separation: float = Field(
        default=10.0, gt=0, description="Minimum angular separation (degrees) from Galactic Bulge"
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check if the coordinate is too close to the Galactic Bulge.

        Parameters
        ----------
        time : Time
            The time (not used for galactic coordinates).
        ephemeris : Ephemeris
            The ephemeris (not used for galactic coordinates).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to the Galactic Bulge).
        """
        # Galactic Bulge coordinates (J2000)
        galactic_bulge = SkyCoord(ra="17h45m40.04s", dec="-29d00m28.1s", frame="icrs")

        # Calculate angular separation
        separation = coordinate.separation(galactic_bulge)

        # Constrain observations closer than the minimum separation
        in_constraint: npt.NDArray[np.bool_] = separation < (self.min_separation * u.deg)

        return in_constraint
