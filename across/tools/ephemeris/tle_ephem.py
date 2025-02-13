from datetime import datetime, timedelta
from typing import Optional, Union

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    TEME,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from sgp4.api import Satrec  # type: ignore[import-untyped]

from ..core.schemas.tle import TLE
from .base import Ephemeris


# Define the radii of the M
class TLEEphemeris(Ephemeris):
    """
    Ephemeris for space objects using TLE data.
    """

    # TLE for calculating LEO satellites
    tle: Optional[TLE] = None

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
        tle: Optional[TLE] = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.tle = tle

    def prepare_data(self) -> None:
        """Calculate ephemeris based on TLE data"""
        # Check if TLE is loaded
        if self.tle is None:
            raise ValueError("No TLE provided")

        # Load in the TLE data
        satellite = Satrec.twoline2rv(self.tle.tle1, self.tle.tle2)

        # Calculate TEME position and velocity for Satellite
        _, temes_p, temes_v = satellite.sgp4_array(self.timestamp.jd1, self.timestamp.jd2)

        # Convert SGP4 TEME data to astropy ITRS SkyCoord
        teme_p = CartesianRepresentation(temes_p.T * u.km)
        teme_v = CartesianDifferential(temes_v.T * u.km / u.s)
        itrs = SkyCoord(teme_p.with_differentials(teme_v), frame=TEME(obstime=self.timestamp)).itrs
        self.earth_location = itrs.earth_location

        # Calculate satellite position in GCRS coordinate system vector as
        # array of x,y,z vectors in units of km, and velocity vector as array
        # of x,y,z vectors in units of km/s
        self.gcrs = itrs.transform_to(GCRS)


def compute_tle_ephemeris(
    begin: Union[datetime, Time],
    end: Union[datetime, Time],
    step_size: Union[int, timedelta, TimeDelta],
    tle: TLE,
) -> Ephemeris:
    """
    Compute the ephemeris for a space object using Two-Line Element (TLE) data.

    Parameters
    ----------
    begin : Union[datetime, Time]
        The start time for the ephemeris computation.
    end : Union[datetime, Time]
        The end time for the ephemeris computation.
    step_size : Union[int, timedelta, TimeDelta]
        Time step size between ephemeris points, in seconds if int
    tle : TLE
        The TLE data entry for the space object.

    Returns
    -------
    Ephemeris
        The computed ephemeris object containing the position and velocity data.
    """
    # Compute the ephemeris using the TLEEphemeris class
    ephemeris = TLEEphemeris(tle=tle, begin=begin, end=end, step_size=step_size)
    ephemeris.compute()
    return ephemeris
