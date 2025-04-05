from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import astroquery.jplhorizons as jpl  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    ITRS,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from .base import Ephemeris


class JPLEphemeris(Ephemeris):
    """
    JPL Horizons-based ephemeris calculation.

    This class provides functionality to calculate ephemeris data using JPL
    Horizons system. It requires a Navigation and Ancillary Information
    Facility (NAIF) ID to identify the celestial object of interest.

    Parameters
    ----------
    begin : datetime or Time
        Start time of the ephemeris calculation
    end : datetime or Time
        End time of the ephemeris calculation
    step_size : int or TimeDelta or timedelta, default=60
        Time step between ephemeris points in seconds
    naif_id : int, optional
        NAIF ID of the celestial object for JPL Horizons query

    Attributes
    ----------
    naif_id : int
        NAIF ID of object for JPL Horizons or Spice Kernel
    gcrs : SkyCoord
        Geocentric celestial reference system coordinates
    earth_location : EarthLocation
        Earth-fixed coordinates of the object

    Methods
    -------
    prepare_data()
        Calculate ephemeris based on JPL Horizons data

    Notes
    -----
    The class uses the JPL Horizons API to fetch vector data for the specified
    celestial object and converts it to both GCRS and ITRS coordinate frames.

    Raises
    ------
    ValueError
        If no NAIF ID is provided when preparing the data
    """

    # NAIF ID of object for JPL Horizons or Spice Kernel
    naif_id: int | None = None

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
        naif_id: int | None = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.naif_id = naif_id

    def prepare_data(self) -> None:
        """Calculate ephemeris based on JPL Horizons data."""
        # Check that parameters needed for JPL Horizons are set
        if self.naif_id is None:
            raise ValueError("No NAIF ID provided")

        # Calculate the number of steps between start and stop
        num_steps = len(self.timestamp) - 1

        # Create a time range dictionary for Horizons
        horizons_range = {
            "start": str(self.begin.tdb.datetime),
            "stop": str(self.end.tdb.datetime),
            "step": str(num_steps),
        }

        # Fetch the ephemeris vector data from Horizons
        horizons_ephemeris = jpl.Horizons(
            id=self.naif_id,
            location="500@399",
            epochs=horizons_range,
            id_type=None,
        )
        horizons_vectors = horizons_ephemeris.vectors(refplane="earth")

        # Create a GCRS SkyCoord object from the ephemeris data
        self.gcrs = SkyCoord(
            CartesianRepresentation(
                horizons_vectors["x"].to(u.km),
                horizons_vectors["y"].to(u.km),
                horizons_vectors["z"].to(u.km),
            ).with_differentials(
                CartesianDifferential(
                    horizons_vectors["vx"],
                    horizons_vectors["vy"],
                    horizons_vectors["vz"],
                )
            ),
            frame=GCRS(obstime=self.timestamp),
        )

        # Calculate the ITRS coordinates and Earth Location
        itrs = self.gcrs.transform_to(ITRS(obstime=self.timestamp))
        self.earth_location = itrs.earth_location


def compute_jpl_ephemeris(
    begin: datetime | Time,
    end: datetime | Time,
    step_size: int | timedelta | TimeDelta,
    naif_id: int,
) -> Ephemeris:
    """
    Compute space ephemeris data using JPL Horizons system.

    Parameters
    ----------
    begin : Union[datetime, Time]
        Start date and time for ephemeris computation
    end : Union[datetime, Time]
        End date and time for ephemeris computation
    step_size : Union[int, timedelta, TimeDelta]
        Time step size between ephemeris points, in seconds if int
    naif_id : int
        Navigation and Ancillary Information Facility (NAIF) object identifier
        (e.g., 301 for Moon. -48 for HST)

    Returns
    -------
    Ephemeris
        An Ephemeris object containing the computed ephemeris data

    Notes
    -----
    This function uses the JPL Horizons system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephemeris = compute_jpl_ephemeris(begin, end, 60, 301)
    """
    # Compute the ephemeris using the JPLEphemeris class
    ephemeris = JPLEphemeris(naif_id=naif_id, begin=begin, end=end, step_size=step_size)
    ephemeris.compute()
    return ephemeris
