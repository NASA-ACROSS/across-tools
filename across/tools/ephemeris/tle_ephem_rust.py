from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    ITRS,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from tle_to_gcrs import Ephemeris as RustEphemeris  # type: ignore[import-untyped]

from ..core.schemas.tle import TLE
from .base import Ephemeris


class RustTLEEphemeris(Ephemeris):
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

        # Compute the Ephemeris using the Rust implementation
        rust_ephem = RustEphemeris(
            tle1=self.tle.tle1,
            tle2=self.tle.tle2,
            begin=self.begin.utc.datetime,
            end=self.end.utc.datetime,
            step_size=int(self.step_size.to_value(u.s)),
        )

        # Calculate convert the ITRS coordinates to SkyCoord objects
        self.itrs = SkyCoord(
            CartesianRepresentation(rust_ephem.itrs.position.T * u.km).with_differentials(
                CartesianDifferential(rust_ephem.itrs.velocity.T * u.km / u.s)
            ),
            frame=ITRS(obstime=Time(rust_ephem.timestamp)),
        )

        self.earth_location = self.itrs.earth_location
        self.latitude = self.itrs.earth_location.lat
        self.longitude = self.itrs.earth_location.lon
        self.height = self.itrs.earth_location.height

        # Calculate satellite position in GCRS coordinate system vector as
        # array of x,y,z vectors in units of km, and velocity vector as array
        # of x,y,z vectors in units of km/s
        self.gcrs = SkyCoord(
            CartesianRepresentation(rust_ephem.gcrs.position.T * u.km).with_differentials(
                CartesianDifferential(rust_ephem.gcrs.velocity.T * u.km / u.s)
            ),
            frame=GCRS(obstime=Time(rust_ephem.timestamp)),
        )

        # Compute gcrs_frame for sun and moon calculations
        # (observer info and obstime)
        gcrs_frame = GCRS(
            obstime=self.timestamp,
            obsgeoloc=rust_ephem.gcrs.position.T * u.km,
            obsgeovel=rust_ephem.gcrs.velocity.T * (u.km / u.s),
        )

        # Calculate the position of the Sun relative to the spacecraft
        sun_pos = CartesianRepresentation(rust_ephem.sun.position.T * u.m)
        sun_vel = CartesianDifferential(rust_ephem.sun.velocity.T * (u.m / u.s))

        # Combine into a full 6D representation
        sun_rep = sun_pos.with_differentials(sun_vel)

        # Create a GCRS frame with observer info and obstime
        self.sun = SkyCoord(sun_rep, frame=gcrs_frame)

        # Calculate the position of the Moon relative to the spacecraft
        moon_pos = CartesianRepresentation(rust_ephem.moon.position.T * u.m)
        moon_vel = CartesianDifferential(rust_ephem.moon.velocity.T * (u.m / u.s))

        # Combine into a full 6D representation,
        moon_rep = moon_pos.with_differentials(moon_vel)

        # Create a GCRS frame with observer info and obstime
        self.moon = SkyCoord(moon_rep, frame=gcrs_frame)

        earth_pos = CartesianRepresentation(-rust_ephem.gcrs.position.T * u.m)
        earth_vel = CartesianDifferential(-rust_ephem.gcrs.velocity.T * (u.m / u.s))
        earth_rep = earth_pos.with_differentials(earth_vel)
        self.earth = SkyCoord(earth_rep, frame=gcrs_frame)


def compute_rust_tle_ephemeris(
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
    ephemeris = RustTLEEphemeris(tle=tle, begin=begin, end=end, step_size=step_size)
    ephemeris.compute()
    return ephemeris
