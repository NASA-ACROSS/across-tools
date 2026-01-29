from collections import OrderedDict
from uuid import UUID

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field, field_validator

from ..core.enums import ConstraintType
from ..ephemeris.base import Ephemeris
from .base import Visibility
from .constraints.base import ConstraintABC


class EphemerisVisibility(Visibility):
    """
    A class for calculating visibility windows based on ephemeris data and constraints.
    This class extends the base Visibility class to compute visibility periods using
    ephemeris data and multiple constraints. It processes time series data to determine
    when specified constraints are met and generates visibility windows accordingly.

    Parameters
    ----------
    constraints : ObservatoryConstraints
        List of constraint objects to be evaluated
    timestamp : Time
        Array of time points for visibility calculations
    calculated_constraints : dict[str, np.typing.NDArray[np.bool_]]
        Dictionary mapping constraint names to boolean arrays of evaluation results
    inconstraint : np.typing.NDArray[np.bool_]
        Boolean array indicating combined constraint evaluation results
    ephemeris : Ephemeris | None
        Ephemeris data object containing spacecraft position/timing information
    step_size : int
        Time step size in seconds for calculations (60s for high res, 3600s for
        low res)

    Methods
    -------
    get_ephemeris_vis()
        Calculates visibility windows based on ephemeris data and constraints
    constraint(index)
        Determines which constraint is active at a given time index
    make_windows(inconstraint)
        Generates visibility window objects from boolean constraint data

    Notes
    -----
    The class processes ephemeris data against multiple constraints to determine
    periods of visibility. It handles both high and low resolution timing and
    generates windows with start/end times and constraint information.
    """

    ephemeris: Ephemeris = Field(..., exclude=True)
    constraints: list[ConstraintABC] = Field(default_factory=list)

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, v: ConstraintABC | list[ConstraintABC]) -> list[ConstraintABC]:
        """Normalize single constraint to list."""
        if isinstance(v, list):
            return v
        return [v]

    def prepare_data(self) -> None:
        """
        Query visibility for given parameters.

        Returns
        -------
            True if successful, False otherwise.
        """
        # Calculate all the individual constraints
        self.calculated_constraints = OrderedDict()

        # If we have composite constraints, we need to handle them differently
        # For now, just evaluate each constraint as a separate item
        for constraint in self.constraints:  # FIXME: constraints constraints
            self.calculated_constraints[constraint.name] = constraint(
                time=self.timestamp, ephemeris=self.ephemeris, coordinate=self.coordinate
            )

        # self.inconstraint is the logical or of all constraints
        # This works for both simple and composite constraints since they all
        # return boolean arrays when called
        if self.calculated_constraints:
            self.inconstraint = np.logical_or.reduce([v for v in self.calculated_constraints.values()])
        else:
            if self.timestamp is None:
                raise ValueError("Timestamp not computed. Call _compute_timestamp() first.")
            self.inconstraint = np.zeros(len(self.timestamp), dtype=np.bool_)

        # Calculate good windows from combined constraints
        self.visibility_windows = self._make_windows()

    def _compute_timestamp(self) -> None:
        """
        Compute timestamp array for visibility calculation.

        This method is called to ensure that the timestamp array is set before
        any visibility calculations are performed.
        """
        if self.ephemeris is None:
            raise ValueError("Ephemeris not available for timestamp computation.")
        # Use datetimes based on the ephemeris calculated timestamps
        self.timestamp = self.ephemeris.timestamp[
            self.ephemeris.index(self.begin) : self.ephemeris.index(self.end)
        ]

    def _find_violated_constraint(self, constraint: ConstraintABC, index: int) -> ConstraintType:
        """
        Find which actual constraint is violated at a given index.

        For logical constraints (And/Or/Not/Xor), recursively check sub-constraints
        to find the actual constraint that caused the violation.

        Parameters
        ----------
        constraint
            The constraint to check (may be logical or regular)
        index
            Index of timestamp to check

        Returns
        -------
            The ConstraintType of the violated constraint
        """
        from .constraints.logical import AndConstraint, NotConstraint, OrConstraint, XorConstraint

        if self.timestamp is None or self.ephemeris is None:
            return ConstraintType.UNKNOWN

        # If it's a logical constraint, drill down to find the actual violated constraint
        if isinstance(constraint, OrConstraint):
            # For OR: any sub-constraint that is violated
            for sub_constraint in constraint.constraints:
                sub_result = sub_constraint(
                    time=self.timestamp[index : index + 1],
                    ephemeris=self.ephemeris,
                    coordinate=self.coordinate,
                )
                if sub_result[0]:  # If this constraint is violated
                    return self._find_violated_constraint(sub_constraint, index)
            return ConstraintType.UNKNOWN

        elif isinstance(constraint, AndConstraint):
            # For AND: all sub-constraints must be violated, return first one
            for sub_constraint in constraint.constraints:
                sub_result = sub_constraint(
                    time=self.timestamp[index : index + 1],
                    ephemeris=self.ephemeris,
                    coordinate=self.coordinate,
                )
                if sub_result[0]:  # If this constraint is violated
                    return self._find_violated_constraint(sub_constraint, index)
            return ConstraintType.UNKNOWN

        elif isinstance(constraint, NotConstraint):
            # For NOT: drill down to the wrapped constraint
            return self._find_violated_constraint(constraint.constraint, index)

        elif isinstance(constraint, XorConstraint):
            # For XOR: find first sub-constraint that is violated
            for sub_constraint in constraint.constraints:
                sub_result = sub_constraint(
                    time=self.timestamp[index : index + 1],
                    ephemeris=self.ephemeris,
                    coordinate=self.coordinate,
                )
                if sub_result[0]:  # If this constraint is violated
                    return self._find_violated_constraint(sub_constraint, index)
            return ConstraintType.UNKNOWN

        else:
            # It's a regular constraint, return its type
            return constraint.name

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

        # Find which constraint is violated at this index
        for constraint_type, violation_array in self.calculated_constraints.items():
            if violation_array[index]:
                # Find the actual constraint object
                matching_constraint = next(
                    (c for c in self.constraints if c.name == constraint_type),
                    None,
                )
                if matching_constraint:
                    return self._find_violated_constraint(matching_constraint, index)
                return constraint_type

        return ConstraintType.UNKNOWN


def compute_ephemeris_visibility(
    begin: Time,
    end: Time,
    ephemeris: Ephemeris,
    constraints: list[ConstraintABC],
    ra: float | None = None,
    dec: float | None = None,
    coordinate: SkyCoord | None = None,
    step_size: u.Quantity = 60 * u.s,
    observatory_name: str = "Observatory",
    observatory_id: UUID | None = None,
    min_vis: int = 0,
) -> EphemerisVisibility:
    """
    Compute visibility windows based on ephemeris data and constraints.

    Parameters
    ----------
    ra: float | None
        Right Ascension in degrees, if applicable.
    dec: float | None
        Declination in degrees, if applicable.
    coordinate: SkyCoord | None
        SkyCoord object representing the position in the sky, if applicable.
    ephemeris : Ephemeris
        The ephemeris data to use for visibility calculations.
    constraints : list[Constraint]
        List of constraints to apply for visibility calculations.
    begin : Time
        Start time for visibility calculation.
    end : Time
        End time for visibility calculation.
    step_size : u.Quantity, optional
        Step size for the timestamp array, default is 60 seconds.
    observatory_name : str, optional
        Name of the observatory for which visibility is calculated, default is "Observatory".
    observatory_id : UUID, optional
        Unique identifier for the observatory, if available.
    min_vis : int, optional
        Minimum visibility time for a window to be considered valid, default is 0 seconds.

    Returns
    -------
    EphemerisVisibility
        An instance of EphemerisVisibility with computed visibility windows.
    """
    vis = EphemerisVisibility(
        ra=ra,
        dec=dec,
        coordinate=coordinate,
        ephemeris=ephemeris,
        constraints=constraints,
        begin=begin,
        end=end,
        step_size=step_size,
        observatory_name=observatory_name,
        min_vis=min_vis,
    )
    if observatory_id is not None:
        vis.observatory_id = observatory_id
    vis.compute()
    return vis
