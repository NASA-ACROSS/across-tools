from datetime import datetime, timedelta
from typing import Optional, Union

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    EarthLocation,
    Latitude,
    Longitude,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from .base import Ephemeris


# Define the radii of the M
class GroundEphemeris(Ephemeris):
    """
    Ephemeris for ground-based observations.
    """

    # Longitude of Observatory on Earth
    longitude: Optional[Longitude]
    # Latitude of Observatory on Earth
    latitude: Optional[Latitude]
    # Height of the Observatory on Earth
    height: Optional[u.Quantity]

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
        latitude: Optional[Latitude] = None,
        longitude: Optional[Longitude] = None,
        height: Optional[u.Quantity] = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.latitude = latitude
        self.longitude = longitude
        self.height = height

    def prepare_data(self) -> None:
        """Calculate ground-based ephemeris"""
        # Check if location of observatory is set.
        if self.latitude is None or self.longitude is None or self.height is None:
            raise ValueError("Location of observatory not set")

        # Set Earth Location based on latitude, longitude, and height
        self.earth_location = EarthLocation.from_geodetic(
            lat=self.latitude, lon=self.longitude, height=self.height
        )

        # Calculate GCRS coordinates of the observatory
        self.gcrs = self.earth_location.get_gcrs(self.timestamp)


def compute_ground_ephemeris(
    begin: Union[datetime, Time],
    end: Union[datetime, Time],
    step_size: Union[int, timedelta, TimeDelta],
    latitude: Latitude,
    longitude: Longitude,
    height: u.Quantity,
) -> Ephemeris:
    """
    Compute ground-based ephemeris for a given time range and location.

    Parameters
    ----------
    begin : Union[datetime, Time]
        The start time of the ephemeris computation.
    end : Union[datetime, Time]
        The end time of the ephemeris computation.
    step_size : int
        The step size in seconds for the ephemeris computation.
    latitude : Latitude
        The latitude of the ground-based observatory.
    longitude : Longitude
        The longitude of the ground-based observatory.
    height : u.Quantity
        The height of the ground-based observatory above sea level.

    Returns
    -------
    Ephemeris
        An Ephemeris object containing the computed ephemeris data.
    """
    # Compute the ephemeris using the GroundEphemeris class
    ephemeris = GroundEphemeris(
        begin=begin, end=end, step_size=step_size, latitude=latitude, longitude=longitude, height=height
    )
    ephemeris.compute()

    return ephemeris
