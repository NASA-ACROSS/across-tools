from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import rust_ephem
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from ..core.schemas.tle import TLE
from .base import Ephemeris


class TLEEphemeris(Ephemeris):
    """
    TLE (Two-Line Element) based satellite ephemeris calculator.

    This class implements satellite position and velocity calculations using
    TLE data through the SGP4 propagator. It converts TEME (True Equator Mean
    Equinox) coordinates to ITRS (International Terrestrial Reference System)
    and GCRS (Geocentric Celestial Reference System). ITRS coordinates are
    stored as astropy EarthLocation objects, while GCRS coordinates are stored as
    astropy SkyCoord objects.

    Parameters
    ----------
    begin : Union[datetime, Time]
        Start time for ephemeris calculations
    end : Union[datetime, Time]
        End time for ephemeris calculations
    step_size : Union[int, TimeDelta, timedelta], optional
        Time step between calculations, defaults to 60 seconds
    tle : Optional[TLE], optional
        TLE data object containing orbital elements, defaults to None

    Attributes
    ----------
    tle : Optional[TLE]
        TLE data object used for calculations
    earth_location : EarthLocation
        Satellite positions in ITRS coordinates
    gcrs : SkyCoord
        Satellite positions in GCRS coordinates

    Methods
    -------
    prepare_data()
        Calculates satellite ephemeris using TLE data and SGP4 propagator

    Raises
    ------
    ValueError
        If no TLE data is provided when preparing calculations

    Notes
    -----
    The calculation process involves:
    1. Converting TLE to Satrec object
    2. Calculating TEME positions using SGP4
    3. Converting TEME to ITRS coordinates
    4. Converting ITRS to GCRS coordinates
    """

    # TLE for calculating LEO satellites
    tle: TLE | None = None

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
        tle: TLE | None = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.tle = tle

    def prepare_data(self) -> None:
        """Calculate ephemeris based on TLE data"""
        # Check if TLE is loaded
        if self.tle is None:
            raise ValueError("No TLE provided")

        # Load in the TLE data
        tle_ephem = rust_ephem.TLEEphemeris(
            begin=self.begin.datetime if isinstance(self.begin, Time) else self.begin,
            end=self.end.datetime if isinstance(self.end, Time) else self.end,
            step_size=int(self.step_size.to_value(u.s))
            if isinstance(self.step_size, (TimeDelta, u.Quantity))
            else int(self.step_size),
            tle1=self.tle.tle1,
            tle2=self.tle.tle2,
        )

        # Calculate satellite position in ITRS coordinate system
        self.earth_location = tle_ephem.itrs.earth_location
        self.latitude = tle_ephem.latitude_deg * u.deg
        self.longitude = tle_ephem.longitude_deg * u.deg
        self.height = tle_ephem.height_km * u.km

        # Calculate satellite position in GCRS coordinate system vector as
        # array of x,y,z vectors in units of km, and velocity vector as array
        # of x,y,z vectors in units of km/s
        self.gcrs = tle_ephem.gcrs


def compute_tle_ephemeris(
    begin: datetime | Time,
    end: datetime | Time,
    step_size: int | timedelta | TimeDelta,
    tle: TLE,
) -> Ephemeris:
    """
    Compute the ephemeris for a space object using TLE data.

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
