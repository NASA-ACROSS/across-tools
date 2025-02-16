from functools import cached_property
from typing import Optional

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ..ephemeris.base import Ephemeris
from .base import Visibility
from .constraints.base import Constraint
from .schema import VisWindow


class EphemerisVisibility(Visibility):
    """
    Calculate visibility of a given object, based on a given spacecraft
    Ephemeris and constraints. Currently supported constraints: Sun, Moon,
    Earth Limb, SAA, Pole.

    Parameters
    ----------
    ra
        Right Ascension in decimal degrees
    dec
        Declination in decimal degrees
    begin
        Start time of visibility search
    end
        End time of visibility search
    hires
        Use high resolution ephemeris for calculations and include earth
        occultations
    constraints
        ObservatoryConstraints object to use for visibility calculations

    Attributes
    ----------
    ephemeris
        Ephemeris object to use for visibility calculations
    saa
        SAA object to use for visibility calculations
    entries
        list of visibility windows
    timestamp
        Array of timestamps
    calculated_constraints
        dictionary of calculated constraints
    inconstraint
        Array of booleans indicating if the spacecraft is in a constraint

    Methods
    -------
    get_ephemeris_vis
        Query visibility for given parameters
    constraint
        What kind of constraints are in place at a given time index
    make_windows
        Create a list of visibility windows from constraint arrays

    """

    # Constraint definitions
    constraints: list[Constraint]

    # Computed values
    timestamp: Time = Field(np.array([]), exclude=True)
    calculated_constraints: dict[str, np.typing.NDArray[np.bool_]] = Field({}, exclude=True)
    inconstraint: np.typing.NDArray[np.bool_] = Field(np.array([]), exclude=True)
    ephemeris: Optional[Ephemeris] = Field(None, exclude=True)

    @cached_property
    def ephstart(self) -> Optional[int]:
        """
        Returns the ephemeris index of the beginning time.
        """
        if self.ephemeris is None:
            return None
        return self.ephemeris.index(Time(self.begin))

    @cached_property
    def ephstop(self) -> Optional[int]:
        """
        Returns the ephemeris index of the stopping time.
        """
        if self.ephemeris is None:
            return None
        i = self.ephemeris.index(Time(self.end))
        if i is None:
            return None
        return i + 1

    @property
    def step_size(self) -> int:
        """
        Returns the step size in seconds based on the hires attribute.
        If high resolution is enabled, returns 60 seconds, otherwise returns 3600 seconds.
        """
        if self.hires:
            return 60
        return 3600

    async def get_ephemeris_vis(self) -> bool:
        """
        Query visibility for given parameters.

        Returns
        -------
            True if successful, False otherwise.
        """
        # Reset windows
        self.entries = list()

        # Check if constraints are available
        if self.constraints is None:
            raise ValueError("Constraints not available.")

        # Check everything is kosher, if just run calculation
        if (
            self.ephemeris is None
            or self.ephemeris.timestamp is None
            or self.ephstart is None
            or self.ephstop is None
        ):
            raise ValueError("Ephemeris not available.")

        # Calculate the times to calculate the visibility
        self.timestamp = self.ephemeris.timestamp[self.ephstart : self.ephstop]

        # Check constraints are array-like
        for con in self.constraints:
            if not isinstance(con, (np.typing.NDArray)):
                raise ValueError("Constraints must be an array-like structure.")

        # Calculate all the individual constraints
        self.calculated_constraints = {
            constraint.short_name: constraint(
                time=self.timestamp, ephemeris=self.ephemeris, skycoord=self.skycoord
            )
            for constraint in self.constraints
        }

        # self.inconstraint is the logical or of all constraints
        self.inconstraint = np.logical_or.reduce([v for v in self.calculated_constraints.values()])

        # Calculate good windows from combined constraints
        self.entries = await self.make_windows(self.inconstraint)

        return True

    async def constraint(self, index: int) -> str:
        """
        What kind of constraints are in place at a given time index.

        Parameters
        ----------
        index
            Index of timestamp to check

        Returns
        -------
            String indicating what constraint is in place at given time index
        """
        # Sanity check
        assert self.ephstart is not None
        assert self.ephstop is not None

        # Check if index is out of bounds
        if index == self.ephstart - 1 or index >= self.ephstop - 1:
            return "Window"

        # Return what constraint is causing the window to open/close
        for k, v in self.calculated_constraints.items():
            if v[index]:
                return k

        return "Unknown"

    async def make_windows(self, inconstraint: np.typing.NDArray[np.bool_]) -> list[VisWindow]:
        """
        Record SAAEntry from array of booleans and timestamps

        Parameters
        ----------
        inconstraint : list
            list of booleans indicating if the spacecraft is in the SAA

        Returns
        -------
        list
            list of SAAEntry objects
        """

        # Find the start and end of the visibility windows
        buff: np.typing.NDArray[np.bool_] = np.concatenate(([False], np.logical_not(inconstraint), [False]))
        begin = np.flatnonzero(~buff[:-1] & buff[1:])
        end = np.flatnonzero(buff[:-1] & ~buff[1:])
        indices = np.column_stack((begin, end - 1))

        # Return as list of VisWindows
        return [
            VisWindow(
                begin=self.timestamp[i[0]].datetime,
                end=self.timestamp[i[1]].datetime,
                visibility=int((self.timestamp[i[1]] - self.timestamp[i[0]]).to_value(u.s)),
                initial=await self.constraint(i[0] - 1),
                final=await self.constraint(i[1] + 1),
            )
            for i in indices
            if self.timestamp[i[0]].datetime != self.timestamp[i[1]].datetime
        ]
