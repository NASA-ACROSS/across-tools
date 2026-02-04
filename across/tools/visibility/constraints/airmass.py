from typing import Literal

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ...core.enums import ConstraintType
from ...ephemeris import Ephemeris
from .base import ConstraintABC, get_slice


class AirmassConstraint(ConstraintABC):
    """
    Constraint based on airmass (zenith angle) for ground-based telescopes.

    Airmass affects the quality of astronomical observations. Higher airmass
    means more atmospheric absorption and poorer image quality. This constraint
    ensures observations are conducted at acceptable airmass values.

    Parameters
    ----------
    max_air_mass : float
        Maximum allowed airmass. Observations with higher airmass will be constrained.
        Typical values: 1.5-2.0 for good quality, up to 3.0 for some surveys.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the airmass is too high for the given coordinate and time.
    """

    name: Literal[ConstraintType.AIRMASS] = ConstraintType.AIRMASS
    short_name: Literal["Airmass"] = "Airmass"
    max_air_mass: float = Field(default=2.0, gt=1.0, description="Maximum allowed airmass for observations")

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check if the airmass is too high for observation.

        Parameters
        ----------
        time : Time
            The time of observation.
        ephemeris : Ephemeris
            The ephemeris (used to get Earth location).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates the airmass is too high
            (constraint violated).
        """

        # Find the slice of ephemeris data we need
        i = get_slice(time, ephemeris)

        # Convert coordinate to AltAz for the given time and location
        self.computed_values.alt_az = coordinate.transform_to(
            AltAz(obstime=time[i], location=ephemeris.earth_location)
        )

        # Calculate airmass using Kasten-Young formula for high accuracy
        h_deg = self.computed_values.alt_az.alt.to_value(u.deg)

        # Clip altitudes to valid range for airmass calculation to avoid
        # unphysical values when below horizon or at zenith. This also prevents
        # division by zero in the formula.
        h_deg = np.clip(h_deg, 0.0, 90.0)
        self.computed_values.air_mass = 1 / (
            np.sin(np.radians(h_deg)) + 0.50572 * (h_deg + 6.07995) ** (-1.6364)
        )

        # Constrain observations with airmass above the maximum
        in_constraint: npt.NDArray[np.bool_] = self.computed_values.air_mass > self.max_air_mass

        return in_constraint
