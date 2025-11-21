from typing import Any, Generic, TypeVar
from uuid import UUID

import numpy as np
from pydantic import Field, model_validator

from ..core.enums import ConstraintType
from ..core.schemas import (
    AstropyDateTime,
    AstropyTimeDelta,
)
from .base import Visibility

T = TypeVar("T", bound=Visibility)


class JointVisibility(Visibility, Generic[T]):
    """
    Computes joint visibility windows between multiple instruments.

    This class takes a list of Visibility objects with identical timestamp grids
    and computes the intersection of their visibility periods.

    Parameters
    ----------
    visibilities : list[T]
        List of Visibility or Visibility child objects with identical timestamp grids.
    instrument_ids : list[UUID]
        List of IDs of the instruments belonging to the Visibility objects.
    """

    # Parameters
    visibilities: list[T] = Field(default_factory=list, exclude=True)
    instrument_ids: list[UUID] = Field(default_factory=list, exclude=True)

    # Values derived from input parameters
    step_size: AstropyTimeDelta = Field(default=None)
    begin: AstropyDateTime = Field(default=None)
    end: AstropyDateTime = Field(default=None)
    observatory_name: str = Field(default="", exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_parameters(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and synchronize coordinate, begin/end, and step size values.
        This method ensures that all the input Visibility objects have the same
        coordinates, begin and end times, and step sizes. If any of these values
        differ between Visibilities, a ValueError is raised.
        When these equalities have been validated, this method runs the base Visibility
        validate_parameters method to validate other parameter values.
        """
        # Check that all ra/dec values are the same within tolerance (~15 arcsec)
        tolerance = 15.0 / 3600.0

        ras = np.array([(visibility.ra) for visibility in values["visibilities"]], dtype=float)
        decs = np.array([(visibility.dec) for visibility in values["visibilities"]], dtype=float)

        if not (np.allclose(ras, ras[0], atol=tolerance) and np.allclose(decs, decs[0], atol=tolerance)):
            raise ValueError(
                f"All input visibilities must have the same coordinate within {tolerance} degrees"
            )
        values["ra"] = values["visibilities"][0].ra
        values["dec"] = values["visibilities"][0].dec

        # Check that all begin/end values are the same
        if (
            not len(set([visibility.begin for visibility in values["visibilities"]])) == 1
            or not len(set([visibility.end for visibility in values["visibilities"]])) == 1
        ):
            raise ValueError("All begin and end times must be the same")
        values["begin"] = values["visibilities"][0].begin
        values["end"] = values["visibilities"][0].end

        # Check that all step sizes are the same
        if not len(set([visibility.step_size for visibility in values["visibilities"]])) == 1:
            raise ValueError("All step sizes must be the same")
        values["step_size"] = values["visibilities"][0].step_size

        return values

    def _constraint(self, i: int) -> ConstraintType:
        """
        For a given index, return the constraint
        from the first visibility that is actually constrained.
        """
        # Find the first visibility that is constrained at this index
        for vis in self.visibilities:
            if vis.inconstraint[i]:
                return vis._constraint(i)

        return ConstraintType.UNKNOWN

    def _get_id(self, i: int) -> UUID:
        """
        For a given index, find the ID of the first instrument that is constrained.
        """
        for vis, instrument_id in zip(self.visibilities, self.instrument_ids):
            if vis.inconstraint[i]:
                return instrument_id

        # Unknown constraint found, so just return the first instrument ID
        return self.instrument_ids[0]

    def _get_name(self, i: int) -> str:
        """
        For a given index, get the name of the first instrument that is constrained.
        """
        for vis in self.visibilities:
            if vis.inconstraint[i]:
                return vis.observatory_name

        # Unknown constraint found, so just return the first instrument name
        return self.observatory_name

    def prepare_data(self) -> None:
        """
        Compute joint visibility by ANDing all inconstraint arrays.

        Raises
        ------
        ValueError
            If visibilities list is empty or if visibilities have different timestamp grids.
        """
        if not self.visibilities:
            raise ValueError("No visibilities provided for joint visibility calculation")

        # Compute joint visibility by ORing all inconstraint arrays
        self.inconstraint = np.any([vis.inconstraint for vis in self.visibilities], axis=0)


def compute_joint_visibility(
    visibilities: list[T],
    instrument_ids: list[UUID],
) -> JointVisibility[T]:
    """
    Compute joint visibility windows for any number of instrument Visibilities.
    Assumes that the visibilities are in the same order as instrument_ids.

    Parameters
    ----------
    visibilities: list[T]
        List of Visibility objects or children objects of Visibility.
    instrument_ids: list[UUID]
        List of IDs of the instruments belonging to the Visibility objects.

    Returns
    -------
    list[VisibilityWindow]
        List of VisibilityWindows capturing joint visibility between all inputted instruments.
    """
    joint_vis = JointVisibility(
        visibilities=visibilities,
        instrument_ids=instrument_ids,
    )
    joint_vis.compute()
    return joint_vis
