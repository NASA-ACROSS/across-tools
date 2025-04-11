from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    EarthLocation,
    Latitude,
    Longitude,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from .base import Ephemeris


class GroundEphemeris(Ephemeris):
    """
    Ground-based ephemeris calculator.

    This class extends the base Ephemeris class to calculate ephemeris data for ground-based observations
    from a specific location on Earth's surface.

    Parameters
    ----------
    begin : Union[datetime, Time]
        Start time of the ephemeris calculation
    end : Union[datetime, Time]
        End time of the ephemeris calculation
    step_size : Union[int, TimeDelta, timedelta], optional
        Time step between ephemeris points in seconds, by default 60
    latitude : Optional[Latitude], optional
        Latitude of the observatory, by default None
    longitude : Optional[Longitude], optional
        Longitude of the observatory, by default None
    height : Optional[u.Quantity], optional
        Height of the observatory above sea level, by default None

    Attributes
    ----------
    latitude : Optional[Latitude]
        Latitude of the observatory
    longitude : Optional[Longitude]
        Longitude of the observatory
    height : Optional[u.Quantity]
        Height of the observatory
    earth_location : EarthLocation
        Location of the observatory on the Earth in astropy's
        EarthLocation class.
    gcrs : SkyCoord
        Observatory position in Geocentric Celestial Reference System (GCRS)
        coordinates

    Raises
    ------
    ValueError
        If observatory location (latitude, longitude, or height) is not set

    Notes
    -----
    The class calculates the position of a ground-based observatory in GCRS
    coordinates, which is necessary for various astronomical calculations.
    """

    # Longitude of Observatory on Earth
    longitude: Longitude | None
    # Latitude of Observatory on Earth
    latitude: Latitude | None
    # Height of the Observatory on Earth
    height: u.Quantity | None

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
        latitude: Latitude | None = None,
        longitude: Longitude | None = None,
        height: u.Quantity | None = None,
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
    begin: datetime | Time,
    end: datetime | Time,
    step_size: int | timedelta | TimeDelta,
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
