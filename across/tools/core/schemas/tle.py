from datetime import datetime, timedelta

from pydantic import Field, computed_field

from .base import BaseSchema


class TLEEntry(BaseSchema):
    """
    Represents a single TLE entry in the TLE database.

    Parameters
    ----------
    satname
        The name of the satellite from the Satellite Catalog.
    tle1
        The first line of the TLE.
    tle2
        The second line of the TLE.

    Attributes
    ----------
    epoch
    """

    satname: str  # Partition Key
    tle1: str = Field(min_length=69, max_length=69)
    tle2: str = Field(min_length=69, max_length=69)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def epoch(self) -> datetime:
        """
        Calculate the Epoch of the TLE file. See
        https://celestrak.org/columns/v04n03/#FAQ04 for more information on
        how the year / epoch encoding works.

        Returns
        -------
            The calculated epoch of the TLE.
        """
        # Extract epoch from TLE
        tleepoch = self.tle1.split()[3]

        # Convert 2 number year into 4 number year.
        tleyear = int(tleepoch[0:2])
        year = 2000 + tleyear if tleyear < 57 else 1900 + tleyear

        # Convert day of year into float
        day_of_year = float(tleepoch[2:])

        # Return Time epoch
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
