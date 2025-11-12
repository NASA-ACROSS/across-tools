from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import timedelta
from uuid import UUID, uuid4

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from across.tools.core.schemas import AstropyDateTime, AstropyTimeDelta

from ..core.enums.constraint_type import ConstraintType
from ..core.schemas import (
    ConstrainedDate,
    ConstraintReason,
    VisibilityWindow,
    Window,
)
from ..core.schemas.base import BaseSchema


class Visibility(ABC, BaseSchema):
    """
    Abstract base class for visibility calculations.

    Parameters
    ----------
    ra
        Right ascension of the target in degrees.
    dec
        Declination of the target in degrees.
    coordinate
        SkyCoord object representing the position of the target in sky
        coordinates. Optional if ra and dec are provided.
    begin
        Start time of visibility period.
    end
        End time of visibility period.
    min_vis
        Minimum visibility percentage for the target.
    hires
        Whether to use high resolution for visibility calculations.
    observatory_id
        Unique Observatory ID for which this visibility is calculated.
    observatory_name:
        Name of the observatory for which this visibility is calculated.

    Methods
    -------
    visible(t)
        Check if the target is visible at a given time.
    compute()
        Perform visibility calculation.

    """

    # Parameters
    ra: float | None = Field(default=None, ge=0, lt=360)
    dec: float | None = Field(default=None, ge=-90, le=90)
    step_size: AstropyTimeDelta = TimeDelta(60 * u.s)
    coordinate: SkyCoord | None = Field(default=None, exclude=True)
    begin: AstropyDateTime
    end: AstropyDateTime
    min_vis: int = 0
    observatory_id: UUID = Field(default_factory=uuid4, exclude=True)
    observatory_name: str

    # Computed values
    timestamp: AstropyDateTime | None = Field(default=None, exclude=True)
    inconstraint: np.typing.NDArray[np.bool_] = Field(default=np.array([]), exclude=True)
    calculated_constraints: OrderedDict[ConstraintType, np.typing.NDArray[np.bool_]] = Field(
        default_factory=OrderedDict, exclude=True
    )
    visibility_windows: list[VisibilityWindow] = []

    @model_validator(mode="before")
    @classmethod
    def validate_parameters(cls, values: dict[str, float | SkyCoord]) -> dict[str, float | SkyCoord]:
        """
        Validate and synchronize SkyCoord, RA and DEC values.
        This method ensures that both coordinate representations (SkyCoord object and separate RA/DEC values)
        are present and consistent. If one representation is missing, it is
        computed from the other. In addition, it validates the step size and
        ensures it is a positive value. The begin and end times are rounded to
        the nearest step size, to ensure consistency.

        """
        if not isinstance(values, dict):
            values = values.__dict__

        # Check that either coordinate or ra/dec values are present
        if not values.get("coordinate") and values.get("ra") and values.get("dec"):
            values["coordinate"] = SkyCoord(ra=values["ra"] * u.deg, dec=values["dec"] * u.deg)
        elif not values.get("dec") and not values.get("ra") and values.get("coordinate"):
            values["ra"] = values["coordinate"].icrs.ra.deg
            values["dec"] = values["coordinate"].icrs.dec.deg
        else:
            raise ValueError("Must supply either coordinate or ra/dec values")

        # Extract TimeDelta from values
        if values.get("step_size") is None:
            values["step_size"] = cls.model_fields["step_size"].default

        # convert to step_size TimeDelta
        if isinstance(values["step_size"], timedelta):
            values["step_size"] = TimeDelta(values["step_size"])
        elif isinstance(values["step_size"], int):
            values["step_size"] = TimeDelta(values["step_size"] * u.s)

        # Check that step_size is a positive value
        if values["step_size"] < TimeDelta(0 * u.s):
            raise ValueError("Step size must be a positive value")

        # Round begin/end to step_size
        if values.get("begin") is not None and values.get("end") is not None:
            begin = Time(values["begin"]).unix
            end = Time(values["end"]).unix
            step_size = values["step_size"].to_value(u.s)

            values["begin"] = Time(begin // step_size * step_size, format="unix")
            values["end"] = Time((end // step_size) * step_size, format="unix")

        return values

    def visible(self, t: Time) -> bool | np.typing.NDArray[np.bool_]:
        """
        For a given time or array of times, is the target visible?

        Parameters
        ----------
        t
            Time to check

        Returns
        -------
            True if visible, False if not
        """
        if not t.isscalar:
            # Return an array of visibility booleans for multiple times
            result = np.zeros(len(t), dtype=bool)

            for win in self.visibility_windows:
                mask = (t >= win.window.begin.datetime) & (t <= win.window.end.datetime)
                result |= mask

            return result
        else:
            # Return if visible for a single time
            for win in self.visibility_windows:
                if win.window.begin.datetime <= t <= win.window.end.datetime:
                    return True
            return False

    def _compute_timestamp(self) -> None:
        """
        Compute timestamp array for visibility calculation. The base
        implementation is to create a timestamp array from the begin and end
        times with a step size defined by the step_size attribute.
        """
        self.timestamp = Time(
            np.arange(self.begin.jd, self.end.jd, self.step_size.to_value(u.d)), format="jd"
        )

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
        index = int(np.round((t.unix - self.timestamp[0].unix) // (self.step_size.to_value(u.s))))
        assert index >= 0 and index < len(self.timestamp), "Time outside of allowed range"
        return index

    @abstractmethod
    def _constraint(self, i: int) -> ConstraintType:
        """
        For a given index, return the constraint at that time.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Abstract method to perform visibility calculation.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    def _make_windows(self) -> list[VisibilityWindow]:
        """
        Create visibility windows from the inconstraint array.

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

        visibility_windows = []
        for i in indices:
            constrained_date_begin = ConstrainedDate(
                datetime=self.timestamp[i[0]].datetime,
                constraint=self._constraint(i[0] - 1),
                observatory_id=self.observatory_id,
            )
            constrained_date_end = ConstrainedDate(
                datetime=self.timestamp[i[1]].datetime,
                constraint=self._constraint(i[1] + 1),
                observatory_id=self.observatory_id,
            )
            window = Window(begin=constrained_date_begin, end=constrained_date_end)
            visibility = int((self.timestamp[i[1]] - self.timestamp[i[0]]).to_value(u.s))
            constraint_reason = ConstraintReason(
                start_reason=f"{self.observatory_name} {self._constraint(i[0] - 1).value}",
                end_reason=f"{self.observatory_name} {self._constraint(i[1] + 1).value}",
            )

            visibility_window = VisibilityWindow(
                window=window,
                max_visibility_duration=visibility,
                constraint_reason=constraint_reason,
            )
            # Only add windows if they are > min_vis seconds long
            if self.timestamp[i[1]].datetime - self.timestamp[i[0]].datetime > timedelta(
                seconds=self.min_vis
            ):
                visibility_windows.append(visibility_window)

        return visibility_windows

    def compute(self) -> None:
        """
        Perform visibility calculation.
        """
        self._compute_timestamp()
        self.prepare_data()
        self.visibility_windows = self._make_windows()


def compute_joint_visibility(
    visibilities: list[Visibility],
    instrument_ids: list[UUID],
) -> list[VisibilityWindow | None]:
    """
    Compute joint visibility windows between multiple instruments.
    Assumes that the visibilities are in the same order as instrument_ids

    Parameters
    ----------
    visibilities: list[Visibility]
        List of Visibility objects.
    instrument_ids: list[UUID]
        List of IDs of the instruments belonging to the Visibility objects.

    Returns
    -------
    list[VisibilityWindow]
        List of VisibilityWindows capturing joint visibility between all inputted instruments.
    """
    # Flatten the windows into a list
    visibility_windows = []
    for visibility, instrument_id in zip(visibilities, instrument_ids):
        windows = visibility.visibility_windows
        if not len(windows):
            # One of the instruments doesn't have any windows, so no joint visibility by default
            return []
        
        for window in windows:
            visibility_windows.append(
                [
                    instrument_id,
                    window.window.begin.datetime,
                    window.window.begin.constraint,
                    window.window.end.datetime,
                    window.window.end.constraint,
                ]
            )

    # Transform list of windows into a numpy array to sort and filter
    visibility_array = np.asarray(visibility_windows)

    # Sort the array by begin time to get windows in chronological order
    sorted_begin_time_indices = visibility_array[:, 1].argsort()
    visibility_array = visibility_array[sorted_begin_time_indices]

    # Get all the instrument ids
    all_instrument_ids = set(visibility_array[:, 0])
    
    joint_windows = []
    for row in visibility_array:
        current_start_datetime = row[1]
        current_end_datetime = row[3]

        # Filter only those rows with compatible start and end times to overlap with current row
        mask = np.where(
            (visibility_array[:, 1] < current_end_datetime)
            & (visibility_array[:, 3] >= current_start_datetime)
        )[0]
        filtered_arr = visibility_array[mask]
        filtered_arr = filtered_arr[filtered_arr[:, 1].argsort()]

        # Filter out any rows with the same instrument ID as the current row
        # Will only keep the first row, corresponding to the earliest window,
        # in cases of duplicates.
        _, unique_instrument_indices = np.unique(filtered_arr[:, 0], return_index=True)
        filtered_arr = filtered_arr[unique_instrument_indices]

        # Check that we have all the instrument IDs remaining
        filtered_instrument_ids = set(filtered_arr[:, 0])
        if all([id_ in filtered_instrument_ids for id_ in all_instrument_ids]):
            # Get start info from most constrained start time
            # Finds remaining row with latest begin time
            new_window_start_time_row = filtered_arr[np.argmax(filtered_arr[:, 1]), :]
            new_window_start_time = new_window_start_time_row[1]
            new_window_begin_reason = new_window_start_time_row[2]
            new_window_begin_instrument_id = new_window_start_time_row[0]
            
            # Get end info from most constrained end time
            # Finds remaining row with earliest end time
            new_window_end_time_row = filtered_arr[np.argmin(filtered_arr[:, 3]), :]
            new_window_end_time = new_window_end_time_row[3]
            new_window_end_reason = new_window_end_time_row[4]
            new_window_end_instrument_id = new_window_end_time_row[0]
            
            # Check that it's a valid window
            if new_window_end_time > new_window_start_time:
                d = VisibilityWindow.model_validate(
                    {
                        "window": {
                            "begin": {
                                "datetime": new_window_start_time,
                                "constraint": new_window_begin_reason,
                                "observatory_id": new_window_begin_instrument_id
                            },
                            "end": {
                                "datetime": new_window_end_time,
                                "constraint": new_window_end_reason,
                                "observatory_id": new_window_end_instrument_id,
                            },
                        },
                        "max_visibility_duration": int(
                            (new_window_end_time - new_window_start_time).sec
                        ),
                        "constraint_reason": {
                            "start_reason": new_window_begin_reason,
                            "end_reason": new_window_end_reason,
                        }
                    }
                )
                
                if d not in joint_windows:
                    joint_windows.append(d)

    return joint_windows
