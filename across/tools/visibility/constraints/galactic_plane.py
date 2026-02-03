from typing import Literal

import numpy as np
from astropy.coordinates import Galactic, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC


class GalacticPlaneConstraint(ConstraintABC):
    """
    Constraint that avoids observing too close to the Galactic Plane.

    The Galactic Plane contains high densities of stars and dust, making it
    difficult to observe faint sources. This constraint ensures observations
    are conducted at sufficient latitude from the galactic equator.

    Parameters
    ----------
    min_latitude : float
        Minimum galactic latitude (degrees) required for observation.
        Observations closer to the galactic plane than this will be constrained.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate is too close to the galactic plane.
    """

    name: Literal[ConstraintType.GALACTIC_PLANE] = ConstraintType.GALACTIC_PLANE
    short_name: Literal["Galactic Plane"] = "Galactic Plane"
    min_latitude: float = Field(
        default=10.0, ge=0, le=90, description="Minimum galactic latitude (degrees) for valid observations"
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check if the coordinate is too close to the galactic plane.

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
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to the galactic plane).
        """
        # Convert to galactic coordinates
        galactic_coord = coordinate.transform_to(Galactic())

        # Check if the absolute galactic latitude is less than the minimum
        # This creates a "zone of avoidance" around the galactic plane
        in_constraint: np.typing.NDArray[np.bool_] = np.abs(galactic_coord.b.deg) < self.min_latitude
        return in_constraint
