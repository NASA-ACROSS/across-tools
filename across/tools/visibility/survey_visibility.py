from collections import OrderedDict

import numpy as np
from pydantic import Field

from ..core.enums import ConstraintType
from .base import Visibility
from .constraints import PointingConstraint


class SurveyVisibility(Visibility):
    """
    A class for calculating visibility windows based on survey pointings.
    This class extends the base Visibility class to compute visibility periods using
    survey pointing constraints. It processes time series data to determine
    when pointing constraints are met and generates visibility windows accordingly.

    Parameters
    ----------
    pointings : list[Pointing]
        A list of Pointing objects, representing a survey observation
        as a polygon and a period of time.

    Methods
    -------
    prepare_data()
        Constructs a dictionary of constraints from the pointings.
        The target is considered "constrained" by the pointing if its
        coordinates fall outside the pointing footprint polygon during the
        time period of the pointing.
    constraint(index)
        Determines whether the pointing constraint is valid at a given time index
    make_windows(inconstraint)
        Generates visibility window objects from boolean constraint data
    """

    pointings: list[PointingConstraint] = Field(default_factory=list)

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
        if self.timestamp is None:
            raise ValueError("Timestamp not computed. Call prepare_data() first.")

        # Check if index is out of bounds
        if index < 0 or index >= len(self.timestamp):
            return ConstraintType.WINDOW

        # Return a pointing constraint otherwise
        return ConstraintType.POINTING

    def prepare_data(self) -> None:
        """
        Collates pointings and calculates visibility windows.
        Uses the pointings to determine if the target coordinate
        was within a given pointing at a given time, then calculates
        windows using the combination of these constraints.
        Returns
        -------
            None.
        """
        # Calculate all the individual constraints
        pointing_constraints: list[np.typing.NDArray[np.bool_]] = []
        for pointing in self.pointings:
            pointing_constraints.append(pointing(time=self.timestamp, coordinate=self.coordinate))

        # self.inconstraint is the logical and of all constraints
        # Note that this differs from other Visibility child class methods
        self.inconstraint = np.logical_and.reduce([v for v in pointing_constraints])

        self.calculated_constraints = OrderedDict()
        # Massage all pointing constraints into a single constraint
        self.calculated_constraints[ConstraintType.POINTING] = self.inconstraint

        # Calculate good windows from combined constraints
        self.visibility_windows = self._make_windows()
