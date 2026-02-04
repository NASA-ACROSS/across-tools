from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord, get_body  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC, get_slice


class SolarSystemConstraint(ConstraintABC):
    """
    Constraint that avoids observing too close to bright Solar System objects.

    Planets and other bright Solar System objects can contaminate observations
    or cause detector issues. This constraint ensures observations avoid these
    objects by maintaining sufficient angular separation.

    Parameters
    ----------
    min_separation : float
        Minimum angular separation (degrees) required from Solar System objects.
        Observations closer than this will be constrained.
    bodies : list[str]
        List of Solar System bodies to avoid. Defaults to major planets.
        Options include: 'sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter',
        'saturn', 'uranus', 'neptune', 'pluto'

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate is too close to any Solar System object.
    """

    name: Literal[ConstraintType.SOLAR_SYSTEM] = ConstraintType.SOLAR_SYSTEM
    short_name: Literal["Solar System"] = "Solar System"
    min_separation: float = Field(
        default=10.0, gt=0, description="Minimum angular separation (degrees) from Solar System objects"
    )
    bodies: list[str] = Field(
        default_factory=lambda: ["venus", "mars", "jupiter", "saturn"],
        description="List of Solar System bodies to avoid",
    )

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check if the coordinate is too close to any Solar System object.

        Parameters
        ----------
        time : Time
            The time of observation.
        ephemeris : Ephemeris
            The ephemeris (used to get Earth location for body positions).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.typing.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to a Solar System object).
        """
        if ephemeris.earth_location is None:
            raise ValueError("Earth location required for Solar System object positions")

        # Find the slice of ephemeris data we need
        i = get_slice(time, ephemeris)

        in_constraint = np.zeros(len(time[i]), dtype=bool)

        # Check separation from each specified Solar System body
        for body_name in self.bodies:
            try:
                # Initialize body_separation dict if needed
                if self.computed_values.body_coordinates is None:
                    self.computed_values.body_coordinates = {}

                # Get the body's position at the observation time
                self.computed_values.body_coordinates[body_name] = get_body(
                    body_name, time[i], ephemeris.earth_location
                )

                # Initialize body_separation dict if needed
                if self.computed_values.body_separation is None:
                    self.computed_values.body_separation = {}

                # Calculate angular separation (and record it in computed values)
                self.computed_values.body_separation[body_name] = coordinate.separation(
                    self.computed_values.body_coordinates[body_name]
                )

                # Check if too close
                in_constraint |= self.computed_values.body_separation[body_name] < (
                    self.min_separation * u.deg
                )

            except Exception:
                # Skip bodies that can't be calculated (e.g., if ephemeris data unavailable)
                continue

        return in_constraint
