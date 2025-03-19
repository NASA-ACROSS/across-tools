from collections import OrderedDict
from functools import cached_property

import numpy as np
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ..core.schemas.visibility import ConstraintType, VisibilityWindow
from ..ephemeris.base import Ephemeris
from .base import Visibility
from .constraints.base import Constraint


class EphemerisVisibility(Visibility):
    """
    A class for calculating visibility windows based on ephemeris data and constraints.
    This class extends the base Visibility class to compute visibility periods using
    ephemeris data and multiple constraints. It processes time series data to determine
    when specified constraints are met and generates visibility windows accordingly.

    Parameters
    ----------
    constraints : list[Constraint]
        List of constraint objects to be evaluated
    timestamp : Time
        Array of time points for visibility calculations
    calculated_constraints : dict[str, np.typing.NDArray[np.bool_]]
        Dictionary mapping constraint names to boolean arrays of evaluation results
    inconstraint : np.typing.NDArray[np.bool_]
        Boolean array indicating combined constraint evaluation results
    ephemeris : Ephemeris | None
        Ephemeris data object containing spacecraft position/timing information

    Methods
    -------
    get_ephemeris_vis()
        Calculates visibility windows based on ephemeris data and constraints
    constraint(index)
        Determines which constraint is active at a given time index
    make_windows(inconstraint)
        Generates visibility window objects from boolean constraint data

    Properties
    ----------
    step_size : int
        Time step size in seconds for calculations (60s for high res, 3600s for
        low res)
    visibility_windows : list[VisibilityWindow]
        List of visibility window objects

    Notes
    -----
    The class processes ephemeris data against multiple constraints to determine
    periods of visibility. It handles both high and low resolution timing and
    generates windows with start/end times and constraint information.
    """

    ephemeris: Ephemeris | None = Field(None, exclude=True)
    constraints: list[Constraint] = Field([], exclude=True)
    calculated_constraints: OrderedDict[ConstraintType, np.typing.NDArray[np.bool_]] = Field(
        OrderedDict(), exclude=True
    )
    visibility_windows: list[VisibilityWindow] = []

    @cached_property
    def _ephstart(self) -> int | None:
        """
        Returns the ephemeris index of the beginning time.
        """
        if self.ephemeris is None:
            return None
        return self.ephemeris.index(Time(self.begin))

    @cached_property
    def _ephstop(self) -> int | None:
        """
        Returns the ephemeris index of the stopping time.
        """
        if self.ephemeris is None:
            return None
        i = self.ephemeris.index(Time(self.end))
        if i is None:
            return None
        return i + 1

    def prepare_data(self) -> None:
        """
        Query visibility for given parameters.

        Returns
        -------
            True if successful, False otherwise.
        """
        # Reset windows
        self.visibility_windows = list()

        # Check if constraints are available
        if self.constraints is None:
            raise ValueError("Constraints not available.")

        # Check everything is kosher, if just run calculation
        if (
            self.ephemeris is None
            or self.ephemeris.timestamp is None
            or self._ephstart is None
            or self._ephstop is None
        ):
            raise ValueError("Ephemeris not available.")

        # Calculate all the individual constraints
        self.calculated_constraints = OrderedDict()
        for constraint in self.constraints:
            self.calculated_constraints[constraint.short_name] = constraint(
                time=self.timestamp, ephemeris=self.ephemeris, skycoord=self.skycoord
            )

        # self.inconstraint is the logical or of all constraints
        self.inconstraint = np.logical_or.reduce([v for v in self.calculated_constraints.values()])

        # Calculate good windows from combined constraints
        self.visibility_windows = self._make_windows()

    def _constraint(self, index: int) -> ConstraintType:
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
        assert self._ephstart is not None
        assert self._ephstop is not None

        # Check if index is out of bounds
        if index == self._ephstart - 1 or index >= self._ephstop - 1:
            return ConstraintType.WINDOW

        # Return what constraint is causing the window to open/close
        return next((k for k, v in self.calculated_constraints.items() if v[index]), ConstraintType.UNKNOWN)
