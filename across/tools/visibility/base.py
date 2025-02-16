from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from ..core.schemas.base import BaseSchema
from .schema import VisWindow


class Visibility(ABC, BaseSchema):
    """
    Abstract base class for visibility calculations.

    Parameters
    ----------
    ra
        Right ascension of the target in degrees.
    dec
        Declination of the target in degrees.
    skycoord
        SkyCoord object representing the position of the target in sky coordinates.
    begin
        Start time of visibility period.
    end
        End time of visibility period.
    min_vis
        Minimum visibility percentage for the target.
    hires
        Whether to use high resolution for visibility calculations.

    Methods
    -------
    visible(t)
        Check if the target is visible at a given time.
    get()
        Perform visibility calculation.

    """

    # Parameters
    ra: float = Field(ge=0, lt=360)
    dec: float = Field(ge=-90, le=90)
    step_size: TimeDelta = TimeDelta(60 * u.s)
    skycoord: SkyCoord = Field(..., exclude=True)
    begin: Time
    end: Time
    min_vis: int = 0

    # Computed values
    timestamp: Optional[Time] = None
    inconstraint: np.typing.NDArray[np.bool_] = Field(np.array([]), exclude=True)
    constraint_windows: Optional[dict[str, list[VisWindow]]] = None
    entries: list[VisWindow] = []

    @model_validator(mode="before")
    @classmethod
    def validate_skycoord(
        cls, values: dict[str, Union[float, SkyCoord]]
    ) -> dict[str, Union[float, SkyCoord]]:
        """
        Validate and synchronize SkyCoord, RA and DEC values.
        This method ensures that both coordinate representations (SkyCoord object and separate RA/DEC values)
        are present and consistent. If one representation is missing, it is computed from the other.
        """
        if not isinstance(values, dict):
            values = values.__dict__
        if not values.get("skycoord") and values.get("ra") and values.get("dec"):
            values["skycoord"] = SkyCoord(ra=values["ra"] * u.deg, dec=values["dec"] * u.deg)
        if not values.get("ra") and values.get("skycoord"):
            values["ra"] = values["skycoord"].icrs.ra.deg
        if not values.get("dec") and values.get("skycoord"):
            values["dec"] = values["skycoord"].icrs.dec.deg
        return values

    @field_validator("step_size", mode="before")
    @classmethod
    def validate_step_size(cls, v: Union[int, timedelta, TimeDelta]) -> TimeDelta:
        """
        Ensure that step_size is an astropy.time.TimeDelta, that step_size
        is a positive value.
        """
        # convert to TimeDelta
        if isinstance(v, timedelta):
            v = TimeDelta(v)
        elif isinstance(v, int):
            v = TimeDelta(v * u.s)

        if v < TimeDelta(0 * u.s):
            raise ValueError("Step size must be a positive value")
        return v

    @field_validator("begin", "end", mode="before")
    @classmethod
    def validate_time_steps(cls, v: Time, info: Any) -> Time:
        """Ensure begin and end times start on whole number of step_size."""

        # Convert datetimes into astropy Time objects
        if isinstance(v, datetime):
            v = Time(v)

        # Ensure that begin and end times are rounded to the nearest step_size.
        # begin should be rounded down, end should be rounded up, to ensure
        # that the visibility is calculated for the requested time range.
        unix_time = v.unix

        step_size = info.get("step_size", TimeDelta(60 * u.s)).to_value(u.s)
        if unix_time % step_size != 0:
            if info.field_name == "begin":
                rounded = (unix_time // step_size) * step_size  # floor division for begin
            else:  # end
                rounded = ((unix_time // step_size) + 1) * step_size  # ceil for end
            v = Time(rounded, format="unix")
        return v

    def visible(self, t: Time) -> bool:
        """
        For a given time, is the target visible?

        Parameters
        ----------
        t
            Time to check

        Returns
        -------
            True if visible, False if not
        """
        return any(t >= win.begin and t <= win.end for win in self.entries)

    def _compute_timestamp(self) -> None:
        """
        Compute timestamp array for visibility calculation.
        """
        self.timestamp = Time(np.arange(self.begin.unix, self.end.unix, self.step_size), format="unix")

    def index(self, t: Time) -> int:
        """
        For a given time, return the index in the ephemeris.

        Parameters
        ----------
        t
            Time to find the index for

        Returns
        -------
            Index of the nearest time in the ephemeris
        """
        if self.timestamp is None:
            raise ValueError("Timestamp not computed")
        index = int(np.round((t.jd - self.timestamp[0].jd) // (self.step_size.to_value(u.d))))
        assert index >= 0 and index < len(self.timestamp), "Time outside of allowed range"
        return index

    @abstractmethod
    def _constraint(self, i: int) -> str:
        """
        For a given index, return the constraint at that time.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Abstract method to perform visibility calculation.
        """
        raise NotImplementedError

    def _make_windows(self) -> list[VisWindow]:
        """
        Parameters
        ----------
        inconstraint : list
            list of booleans indicating if the spacecraft is in the SAA

        Returns
        -------
        list
            list of SAAEntry objects
        """
        # Check that timestamp is set and constraints
        if self.timestamp is None:
            raise ValueError("Timestamp not set")

        # Find the start and end of the visibility windows
        buff: np.typing.NDArray[np.bool_] = np.concatenate(
            ([False], np.logical_not(self.inconstraint), [False])
        )
        begin = np.flatnonzero(~buff[:-1] & buff[1:])
        end = np.flatnonzero(buff[:-1] & ~buff[1:])
        indices = np.column_stack((begin, end - 1))

        # Return as list of VisWindows
        return [
            VisWindow(
                begin=self.timestamp[i[0]].datetime,
                end=self.timestamp[i[1]].datetime,
                visibility=int((self.timestamp[i[1]] - self.timestamp[i[0]]).to_value(u.s)),
                initial=self._constraint(i[0] - 1),
                final=self._constraint(i[1] + 1),
            )
            for i in indices
            if self.timestamp[i[0]].datetime != self.timestamp[i[1]].datetime
        ]

    def compute(self) -> None:
        """
        Perform visibility calculation.
        """
        self._compute_timestamp()
        self.prepare_data()
        self.entries = self._make_windows()
