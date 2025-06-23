from abc import ABC, abstractmethod

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from ...core.enums import ConstraintType
from ...core.schemas.base import BaseSchema
from ...ephemeris import Ephemeris


def get_slice(time: Time, ephemeris: Ephemeris) -> slice:
    """
    Return a slice for what the part of the ephemeris that we're using.

    Arguments
    ---------
    time : Time
        The time to calculate the slice for
    ephemeris : Ephemeris
        The spacecraft ephemeris

    Returns
    -------
        The slice for the ephemeris
    """
    # Ensure that time is an array-like object
    if time.isscalar:
        raise NotImplementedError("Scalar time not supported")

    # Find the indices for the start and end of the time range and return a
    # slice for that range
    return slice(ephemeris.index(time[0]), ephemeris.index(time[-1]) + 1)


class ConstraintABC(BaseSchema, ABC):
    """
    Base class for constraints. Constraints are used to determine if a given
    coordinate is inside the constraint. This is done by checking if the
    separation between the constraint and the coordinate is less than a given
    value.

    Methods
    -------
    __call__(time, ephemeris, coord)
        Checks if a given coordinate is inside the constraint.
    """

    short_name: str
    name: ConstraintType

    @abstractmethod
    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the constraint.

        Parameters
        ----------
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover
