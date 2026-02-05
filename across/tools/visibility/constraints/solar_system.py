import logging
from typing import Any, Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord, get_body  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field, field_validator

from ...core.enums import ConstraintType, SolarSystemObject
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
        Options include: 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto'.

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
    max_magnitude: float = Field(
        default=5.0, description="Maximum apparent magnitude of Solar System objects to consider"
    )
    bodies: list[SolarSystemObject | str] = Field(
        default=[
            SolarSystemObject.MERCURY,
            SolarSystemObject.VENUS,
            SolarSystemObject.MARS,
            SolarSystemObject.JUPITER,
            SolarSystemObject.SATURN,
        ],
        description="List of Solar System bodies to avoid",
    )

    @field_validator("bodies", mode="before")
    @classmethod
    def validate_bodies(cls, v: Any) -> list[SolarSystemObject | Any]:
        """
        Validate that bodies is a list of SolarSystemObject or strings that
        match SolarSystemObject enum (case insensitive).
        """
        if isinstance(v, list):
            return [SolarSystemObject(body.lower()) if isinstance(body, str) else body for body in v]
        raise ValueError("bodies must be a list of SolarSystemObject or strings")

    def _calculate_body_magnitude(
        self,
        body_name: SolarSystemObject | str,
        body_coord: SkyCoord,
        ephemeris: Ephemeris,
        i: slice,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the apparent magnitude of a solar system body.

        Parameters
        ----------
        body_name : SolarSystemObject | str
            Name of the body ('moon', 'venus', etc.)
        body_coord : SkyCoord
            Position of the body
        ephemeris : Ephemeris
            Ephemeris data
        i : slice
            Slice for the time array

        Returns
        -------
        npt.NDArray[np.float64]
            Apparent magnitude of the body
        """

        if body_name == SolarSystemObject.MERCURY:
            # Mercury magnitude calculation
            # Simplified Mercury magnitude formula
            # Maximum brightness around -1.9
            phase_angle: npt.NDArray[np.float64] = body_coord.separation(ephemeris.sun[i]).deg
            magnitude: npt.NDArray[np.float64] = -1.9 + 0.02 * phase_angle + 3.5e-7 * phase_angle**3
            return magnitude

        elif body_name == SolarSystemObject.VENUS:
            # Venus magnitude calculation
            # Simplified Venus magnitude formula
            # Maximum brightness around -4.7
            phase_angle = body_coord.separation(ephemeris.sun[i]).deg
            magnitude = -4.7 + 0.013 * phase_angle + 4.3e-7 * phase_angle**3
            return magnitude

        elif body_name == SolarSystemObject.MARS:
            # Mars magnitude calculation
            # Mars magnitude depends on distance and phase
            distance_au = body_coord.distance.to(u.AU)
            magnitude = -1.52 + 5 * np.log10(distance_au.value)
            return magnitude

        elif body_name == SolarSystemObject.JUPITER:
            # Jupiter magnitude calculation
            distance_au = body_coord.distance.to(u.AU)
            magnitude = -9.4 + 5 * np.log10(distance_au.value)
            return magnitude

        elif body_name == SolarSystemObject.SATURN:
            # Saturn magnitude calculation
            distance_au = body_coord.distance.to(u.AU)
            magnitude = -8.9 + 5 * np.log10(distance_au.value)
            return magnitude

        elif body_name == SolarSystemObject.URANUS:
            # Uranus magnitude calculation
            distance_au = body_coord.distance.to(u.AU)
            magnitude = 5.5 + 5 * np.log10(distance_au.value)
            return magnitude

        elif body_name == SolarSystemObject.NEPTUNE:
            # Neptune magnitude calculation
            distance_au = body_coord.distance.to(u.AU)
            magnitude = 7.8 + 5 * np.log10(distance_au.value)
            return magnitude

        else:
            # Unknown body, this should unreachable
            raise ValueError(f"Unknown body for magnitude calculation: {body_name}")

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
            # Initialize body_separation dict if needed
            if self.computed_values.body_coordinates is None:
                self.computed_values.body_coordinates = {}

            # Get the body's position at the observation time
            try:
                if hasattr(ephemeris, "_tle_ephem") or False:
                    # Use rust-ephem's performant get_body if available
                    self.computed_values.body_coordinates[body_name] = ephemeris._tle_ephem.get_body(
                        body_name + " barycenter"
                    )[i]
                else:
                    # Fallback to astropy's get_body
                    self.computed_values.body_coordinates[body_name] = get_body(
                        body_name, time[i], ephemeris.earth_location
                    )
            except (KeyError, ValueError) as e:
                logging.warning(f"Could not get position for body {body_name}: {e}")
                continue

            # Initialize body_separation dict if needed
            if self.computed_values.body_separation is None:
                self.computed_values.body_separation = {}

            # Calculate angular separation (and record it in computed
            # values)
            self.computed_values.body_separation[body_name] = coordinate.separation(
                self.computed_values.body_coordinates[body_name].icrs
            )

            # Initialize body_magnitude dict if needed
            if self.computed_values.body_magnitude is None:
                self.computed_values.body_magnitude = {}

            # Calculate apparent magnitude of the body
            self.computed_values.body_magnitude[body_name] = self._calculate_body_magnitude(
                body_name, self.computed_values.body_coordinates[body_name], ephemeris, i
            )

            # Check if too close
            in_constraint |= self.computed_values.body_separation[body_name] < (self.min_separation * u.deg)

        return in_constraint
