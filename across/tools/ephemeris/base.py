from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Union

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.constants import R_earth, R_sun  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    Angle,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    get_body,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

# Define the radii of the Moon (as astropy doesn't)
R_moon = 1737.4 * u.km


class Ephemeris(ABC):
    """
    Abstract base class for calculating and managing ephemeris data.
    """

    # Parameters
    begin: Time  # Start time of ephemeris calculation
    end: Time  # End time of ephemeris calculation
    step_size: TimeDelta  # Step size of ephemeris calculation in seconds

    # Computed values
    timestamp: Time
    gcrs: SkyCoord
    earth_location: EarthLocation
    moon: SkyCoord
    sun: SkyCoord
    earth: SkyCoord
    longitude: Optional[Longitude]
    latitude: Optional[Latitude]
    height: Optional[u.Quantity]
    earth_radius_angle: Angle
    moon_radius_angle: Angle
    sun_radius_angle: Angle
    distance: u.Quantity

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
    ) -> None:
        # Convert start and stop to astropy Time natively
        self.begin = begin if isinstance(begin, Time) else Time(begin)
        self.end = end if isinstance(end, Time) else Time(end)

        # Convert step_size to TimeDelta
        if isinstance(step_size, TimeDelta):
            self.step_size = step_size
        elif isinstance(step_size, timedelta):
            self.step_size = TimeDelta(step_size)
        elif isinstance(step_size, (int, float)):
            self.step_size = TimeDelta(step_size * u.s)

        # Compute range of timestamps
        self.timestamp = self._compute_timestamp()

    def __len__(self) -> int:
        return len(self.timestamp)

    def index(self, t: Time) -> int:
        """
        For a given time, return an index for the nearest time in the
        ephemeris. Note that internally converting from Time to datetime makes
        this run way faster.

        Parameters
        ----------
        t : Time
            The time to find the nearest index for.

        Returns
        -------
        int
            The index of the nearest time in the ephemeris.
        """
        index = int(np.round((t.unix - self.timestamp[0].unix) // (self.step_size.to_value(u.s))))
        assert index >= 0 and index < len(self), "Time outside of ephemeris of range"
        return index

    def _compute_timestamp(self) -> Time:
        """
        Get array of timestamps based on time interval and step size.

        Returns
        -------
        astropy.time.Time
            If begin equals end, returns single timestamp.
            Otherwise returns array of timestamps from begin to end with specified step_size.
        """

        # Create array of timestamps
        if self.begin == self.end:
            return Time([self.begin])

        return Time(
            np.arange(
                self.begin.datetime,
                self.end.datetime + self.step_size.to_datetime(),
                self.step_size.to_datetime(),
            )
        )

    def _calc(self) -> None:
        """
        Calculate ephemeris data based on the coordinates computed by
        prepare_data().
        """
        # Calculate the position of the Moon relative to the spacecraft
        self.moon = get_body("moon", self.timestamp, location=self.earth_location)

        # Calculate the position of the Sun relative to the spacecraft
        self.sun = get_body("sun", self.timestamp, location=self.earth_location)

        # Calculate the position of the Earth relative to the spacecraft
        self.earth = get_body("earth", self.timestamp, location=self.earth_location)

        # Get the longitude, latitude, height, and distance (from center of
        # Earth) of the satellite from the EarthLocation object.
        self.longitude = self.earth_location.lon
        self.latitude = self.earth_location.lat
        self.height = self.earth_location.height
        self.distance = self.gcrs.distance

        # Calculate the Earth angular radius as seen from the spacecraft
        self.earth_radius_angle = np.arcsin(R_earth / self.distance)

        # Similarly calculate the angular radii of the Sun and the Moon
        self.moon_radius_angle = np.arcsin(R_moon / self.moon.distance)
        self.sun_radius_angle = np.arcsin(R_sun / self.sun.distance)

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare data for ephemeris calculation. Abstract method, to be implemented by subclasses.
        """
        raise NotImplementedError("prepare_data method must be implemented by subclass")

    def compute(self) -> None:
        """
        Compute ephemeris data.

        This method orchestrates the computation of ephemeris data by first
        preparing the necessary data and then performing the core ephemeris
        calculations. It is intended to be called after initializing the
        Ephemeris object with the desired time range and step size.
        """
        self.prepare_data()
        self._calc()
