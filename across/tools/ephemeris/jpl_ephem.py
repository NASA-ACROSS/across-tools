from datetime import datetime, timedelta
from typing import Optional, Union

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


# Define the radii of the M
class JPLEphemeris(Ephemeris):
    """
    Ephemeris for space objects using JPL Horizons.
    """

    # NAIF ID of object for JPL Horizons or Spice Kernel
    naif_id: Optional[int] = None

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
        naif_id: Optional[int] = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.naif_id = naif_id

    def prepare_data(self) -> None:
        """Calculate ephemeris based on JPL Horizons data."""
        # Check that parameters needed for JPL Horizons are set
        if self.naif_id is None:
            raise ValueError("No NAIF ID provided")

        # Create a time range dictionary for Horizons
        horizons_range = {
            "start": str(self.begin.tdb),
            "stop": str(self.end.tdb),
            "step": f"{self.step_size.to_value(u.min):.0f}m",
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
        gcrs_p = CartesianRepresentation(
            horizons_vectors["x"].to(u.km),
            horizons_vectors["y"].to(u.km),
            horizons_vectors["z"].to(u.km),
        )
        gcrs_v = CartesianDifferential(
            horizons_vectors["vx"],
            horizons_vectors["vy"],
            horizons_vectors["vz"],
        )
        self.gcrs = SkyCoord(gcrs_p.with_differentials(gcrs_v), frame=GCRS(obstime=self.timestamp))

        # Calculate the ITRS coordinates and Earth Location
        itrs = self.gcrs.transform_to(ITRS(obstime=self.timestamp))
        self.earth_location = itrs.earth_location


def compute_jpl_ephemeris(
    begin: Union[datetime, Time],
    end: Union[datetime, Time],
    step_size: Union[int, timedelta, TimeDelta],
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
        NAIF object identifier (e.g., 301 for Moon. -48 for HST)

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
