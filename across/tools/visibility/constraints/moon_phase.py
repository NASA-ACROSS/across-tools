from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...ephemeris import Ephemeris
from .base import Constraint, get_slice


class MoonPhaseConstraint(Constraint):
    """
    Constraint based on the current phase of the Moon, as observed from the
    given observatory ephemeris.

    Parameters
    ----------
    min_phase
        The minimum moon phase value.
    max_phase
        The maximum moon phase value.
    """

    name: Literal["Moon Phase"] = "Moon Phase"
    short_name: Literal["Moon Phase"] = "Moon Phase"
    min_phase_angle: float | None = None
    max_phase_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Calculate the Moon phase constraint for a given time and ephemeris.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Moon phase.
        skycoord : SkyCoord
            The coordinate to check.

        Returns
        -------
        np.ndarray
            The calculated constraint values as a NumPy array.
        """
        i = get_slice(time, ephemeris)
        assert (
            ephemeris.sun is not None
            and ephemeris.moon is not None
            and ephemeris.gcrs.cartesian.xyz.value is not None
        )
        # Calculate the moon phase angle
        sun_moon = ephemeris.sun.cartesian[i] - ephemeris.moon.cartesian[i]
        sun_moon /= sun_moon.norm()
        observatory_moon = ephemeris.gcrs.cartesian.xyz.value[i] - ephemeris.moon.cartesian[i]
        observatory_moon /= observatory_moon.norm()
        phase_angle = np.arccos(sun_moon.dot(observatory_moon))

        # Set up the constraint array
        in_constraint = np.zeros(len(phase_angle), dtype=bool)

        # Calculate the basic Moon phase constraint
        if self.min_phase_angle is not None:
            in_constraint |= phase_angle > self.min_phase_angle
        if self.max_phase_angle is not None:
            in_constraint |= phase_angle < self.max_phase_angle

        # Return the value as a scalar or array
        return in_constraint[0] if time.isscalar else in_constraint
