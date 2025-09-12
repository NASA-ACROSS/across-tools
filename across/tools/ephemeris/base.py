from abc import ABC, abstractmethod
from datetime import datetime, timedelta

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
    A base class for calculating ephemerides of Astronomical Observatories both
    ground and space-based. This abstract class provides the core functionality
    for computing positions and angular sizes of celestial bodies (Sun, Moon,
    Earth) relative to a spacecraft or observation point. It handles time
    series calculations with specified time intervals and step sizes.

    Parameters
    ----------
    begin : datetime or Time
        Start time of ephemeris calculation
    end : datetime or Time
        End time of ephemeris calculation
    step_size : int, TimeDelta, or timedelta, optional
        Time step between calculations in seconds, by default 60

    Attributes
    ----------
    begin : Time
        Start time of ephemeris calculation
    end : Time
        End time of ephemeris calculation
    step_size : TimeDelta
        Step size of ephemeris calculation
    timestamp : Time
        Array of calculation timestamps
    gcrs : SkyCoord
        Spacecraft position in Geocentric Celestial Reference System (GCRS)
        coordinates
    earth_location : EarthLocation
        Spacecraft position relative to Earth
    moon : SkyCoord
        Moon position relative to spacecraft
    sun : SkyCoord
        Sun position relative to spacecraft
    earth : SkyCoord
        Earth position relative to spacecraft
    longitude : Longitude, optional
        Spacecraft longitude
    latitude : Latitude, optional
        Spacecraft latitude
    height : Quantity, optional
        Spacecraft height above Earth's surface
    earth_radius_angle : Angle
        Angular radius of Earth as seen from spacecraft
    moon_radius_angle : Angle
        Angular radius of Moon as seen from spacecraft
    sun_radius_angle : Angle
        Angular radius of Sun as seen from spacecraft
    distance : Quantity
        Distance from spacecraft to Earth center

    Notes
    -----
    This is an abstract base class that must be subclassed. Subclasses must implement
    the prepare_data() method to set up the spacecraft position before ephemeris
    calculations can be performed.
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
    longitude: Longitude | None = None
    latitude: Latitude | None = None
    height: u.Quantity | None = None
    earth_radius_angle: Angle
    moon_radius_angle: Angle
    sun_radius_angle: Angle
    distance: u.Quantity

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
    ) -> None:
        # Convert begin and end to astropy Time
        self.begin = begin if isinstance(begin, Time) else Time(begin)
        self.end = end if isinstance(end, Time) else Time(end)

        # Convert step_size to TimeDelta
        if isinstance(step_size, TimeDelta):
            self.step_size = step_size
        elif isinstance(step_size, timedelta):
            self.step_size = TimeDelta(step_size)
        elif isinstance(step_size, (int, float)):
            self.step_size = TimeDelta(step_size * u.s)

        # Align begin/end to step grid (floor)
        s = self.step_size.to_value(u.s)
        self.begin = Time((self.begin.unix // s) * s, format="unix")
        self.end = Time((self.end.unix // s) * s, format="unix")

        # Compute range of timestamps
        self.timestamp = self._compute_timestamp()

    def __len__(self) -> int:
        return len(self.timestamp)

    def index(self, t: Time) -> int:
        """
        For a given time, return an index for the nearest time in the
        ephemeris. Note that internally converting from Time to unix makes
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
        return Time(
            np.arange(
                self.begin.unix, self.end.unix + self.step_size.to_value(u.s), self.step_size.to_value(u.s)
            ),
            format="unix",
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

        # Calculate Earth's angular radius from observatory, capped at 90 degrees
        self.earth_radius_angle = np.arcsin(np.minimum(R_earth / self.distance, 1))

        # Similarly calculate the angular radii of the Sun and the Moon, capped at 90 degrees
        self.moon_radius_angle = np.arcsin(np.minimum(R_moon / self.moon.distance, 1))
        self.sun_radius_angle = np.arcsin(np.minimum(R_sun / self.sun.distance, 1))

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare data for ephemeris calculation. Abstract method, to be implemented by subclasses.
        """
        raise NotImplementedError("prepare_data method must be implemented by subclass")  # pragma: no cover

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
