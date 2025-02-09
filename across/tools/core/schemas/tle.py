from datetime import datetime, timedelta
from typing import Optional

from pydantic import Field, computed_field

from .base import BaseSchema


class TLEBase(BaseSchema):
    """
    A base schema representing a Two-Line Element (TLE) set for satellite tracking.
    Parameters
    ----------
    norad_id : int or None
        The NORAD Catalog Number (SATCAT) that uniquely identifies the satellite.
    satellite_name : str or None
        The name or designation of the satellite.
    tle1 : str
        The first line of the TLE set, must be exactly 69 characters long.
        Contains information about satellite epoch, decay rate, etc.
    tle2 : str
        The second line of the TLE set, must be exactly 69 characters long.
        Contains orbital elements like inclination, eccentricity, etc.
    Notes
    -----
    TLE is a data format encoding a set of orbital elements for Earth-orbiting objects.
    Each TLE line must be exactly 69 characters long per NORAD specification.
    """

    norad_id: Optional[int] = None
    satellite_name: Optional[str] = None
    tle1: str = Field(min_length=69, max_length=69)
    tle2: str = Field(min_length=69, max_length=69)


class TLE(TLEBase):
    """
    Two Line Element (TLE) data representation.

    This class represents a Two Line Element Set, which is a data format used to convey
    sets of orbital elements that describe the orbits of Earth-orbiting satellites.

    Parameters
    ----------
    norid : int
        The NORAD Catalog Number (SATCAT) that uniquely identifies the satellite
    satname : str
        The name of the satellite (used as Partition Key)
    tle1 : str
        First line of the TLE, must be exactly 69 characters
    tle2 : str
        Second line of the TLE, must be exactly 69 characters

    Attributes
    ----------
    epoch : datetime
        The epoch timestamp calculated from the TLE data

    Notes
    -----
    TLE format specifications can be found at:
    https://celestrak.org/NORAD/documentation/tle-fmt.php

    Examples
    --------
    >>> tle = TLE(
    ...     satname="ISS (ZARYA)",
    ...     tle1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    ...     tle2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ... )

    satname: str  # Partition Key
    tle1: str = Field(min_length=69, max_length=69)
    tle2: str = Field(min_length=69, max_length=69)

    @computed_field  # type: ignore[prop-decorator]
    @property
    """

    @computed_field
    def epoch(self) -> datetime:
        """
        Calculate the Epoch of the TLE file. See
        https://celestrak.org/columns/v04n03/#FAQ04 for more information on how
        the year / epoch encoding works.
        Returns
        -------
            The calculated epoch of the TLE.
        """
        # Extract year and days from TLE
        tleepoch = self.tle1.split()[3]
        tleyear = int(tleepoch[:2])
        days = float(tleepoch[2:]) - 1

        # Calculate epoch date
        year = 2000 + tleyear if tleyear < 57 else 1900 + tleyear
        return datetime(year, 1, 1) + timedelta(days=days)
