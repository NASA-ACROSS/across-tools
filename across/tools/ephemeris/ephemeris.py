import os
from datetime import datetime, timedelta
from typing import Optional, Union

import astropy.units as u  # type: ignore[import-untyped]
import astroquery.jplhorizons as jpl  # type: ignore[import-untyped]
import numpy as np
import spiceypy as spice  # type: ignore[import-untyped]
from astropy.constants import R_earth, R_sun  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    ITRS,
    TEME,
    Angle,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    get_body,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from astropy.utils.data import download_file  # type: ignore[import-untyped]
from sgp4.api import Satrec  # type: ignore[import-untyped]

from ..core.enums import EphemerisType
from ..core.schemas.tle import TLE

# Define the radii of the Moon (as astropy doesn't)
R_moon = 1737.4 * u.km

NAIF_LEAP_SECONDS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
NAIF_PLANETARY_EPHEMERIS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp"
NAIF_EARTH_ORIENTATION_PARAMETERS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc"
)
SPICE_KERNEL_CACHE_DIR = os.path.expanduser("~/.cache/across/spice")


class Ephemeris:
    """
    A class for calculating and managing ephemeris data for astronomical objects.
    This class provides functionality to compute ephemeris data for ground-based
    observations and space-based objects using various methods including TLE,
    JPL Horizons, and SPICE kernels.

    Parameters
    ----------
    begin : Union[datetime, Time]
        Start time for ephemeris calculations
    end : Union[datetime, Time]
        End time for ephemeris calculations
    step_size : Union[int, timedelta, TimeDelta]
        Time step size in seconds between ephemeris points, defaults to 60 seconds
    naif_id : Optional[int], optional
        NAIF ID code for the object when using JPL Horizons or SPICE kernels
    tle : Optional[TLE], optional
        Two-line element set for satellite ephemeris calculations
    latitude : Optional[Latitude], optional
        Observer latitude for ground-based calculations
    longitude : Optional[Longitude], optional
        Observer longitude for ground-based calculations
    height : u.Quantity, optional
        Observer height for ground-based calculations
    spice_kernel_url : Optional[str], optional
        URL to SPICE kernel file for SPICE-based calculations

    Attributes
    ----------
    timestamp : Time
        Array of time points for ephemeris
    gcrs : SkyCoord
        Observer location in GCRS coordinate frame
    earth_location : EarthLocation
        Observer location relative to Earth's Surface
    moon : SkyCoord
        Moon positions relative to observer
    sun : SkyCoord
        Sun positions relative to observer
    earth : SkyCoord
        Earth positions relative to observer
    longitude : Longitude
        Observer/satellite longitude
    latitude : Latitude
        Observer/satellite latitude
    height : u.Quantity
        Observer/satellite height
    earth_radius_angle : Angle
        Angular radius of Earth
    moon_radius_angle : Angle
        Angular radius of Moon
    sun_radius_angle : Angle
        Angular radius of Sun

    Methods
    -------
    compute(ephemeris_type)
        Main method to compute ephemeris based on specified type
    ephindex(t)
        Get index of nearest ephemeris point for given time

    Notes
    -----
    The class supports four types of ephemeris calculations:
    - Ground-based observations
    - Space-based using TLE
    - Space-based using JPL Horizons
    - Space-based using SPICE kernels
    All calculations are performed using high-precision astronomical algorithms
    and coordinate transformations.
    """

    # Attributes
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

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
        naif_id: Optional[int] = None,
        tle: Optional[TLE] = None,
        latitude: Optional[Latitude] = None,
        longitude: Optional[Longitude] = None,
        height: Optional[u.Quantity] = None,
        spice_kernel_url: Optional[str] = None,
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

        # Rest of the attributes
        self.tle = tle
        self.latitude = latitude
        self.longitude = longitude
        self.height = height
        self.naif_id = naif_id
        self.spice_kernel_url = spice_kernel_url

    def __len__(self) -> int:
        if self.timestamp is None:
            return 0
        return len(self.timestamp)

    def index(self, t: Time) -> int:
        """
        For a given time, return an index for the nearest time in the
        ephemeris. Note that internally converting from Time to datetime makes
        this run way faster.

        Parameters
        ----------
        t
            The time to find the nearest index for.

        Returns
        -------
            The index of the nearest time in the ephemeris.
        """
        if self.timestamp is None:
            raise Exception("Ephemeris not computed")
        index = int(np.round((t.jd - self.timestamp[0].jd) // (self.step_size.to_value(u.d))))
        assert index >= 0 and index < len(self), "Time outside of ephemeris of range"
        return index

    def _ground_ephemeris(self) -> bool:
        """Calculate ground-based ephemeris"""
        # Check if location of observatory is set.
        if self.latitude is None or self.longitude is None or self.height is None:
            raise Exception("Location of observatory not set")

        self.earth_location = EarthLocation.from_geodetic(
            lat=self.latitude, lon=self.longitude, height=self.height
        )

        # Calculate GCRS and ITRS coordinates of the observatory
        self.gcrs = self.earth_location.get_gcrs(self.timestamp)

        return True

    def _tle_ephemeris(self) -> bool:
        """Calculate ephemeris based on TLE data"""
        # Check if TLE is loaded
        if self.tle is None:
            raise Exception("No TLE available for this epoch")

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
        return True

    def _load_spice_kernels(self) -> None:
        """Loading SPICE kernels to calculate SPICE based ephemeris."""
        if self.spice_kernel_url is None:
            raise Exception("No SPICE kernel URL provided")

        # Helper method to load the kernel files after download
        leap_seconds_file = download_file(NAIF_LEAP_SECONDS_URL, cache=True)
        planetary_ephemeris_file = download_file(NAIF_PLANETARY_EPHEMERIS_URL, cache=True)
        earth_params_file = download_file(NAIF_EARTH_ORIENTATION_PARAMETERS_URL, cache=True)
        spice_kernel_file = download_file(self.spice_kernel_url, cache=True)

        # Check if kernels are already loaded
        loaded_kernels = [str(spice.kdata(i, "all")[0]) for i in range(spice.ktotal("all"))]

        # Load local cached kernel files if not already loaded
        if leap_seconds_file not in loaded_kernels:
            spice.furnsh(leap_seconds_file)  # Leap seconds
        if planetary_ephemeris_file not in loaded_kernels:
            spice.furnsh(planetary_ephemeris_file)  # Planetary ephemeris
        if earth_params_file not in loaded_kernels:
            spice.furnsh(earth_params_file)  # High-precision Earth orientation
        if spice_kernel_file not in loaded_kernels:
            spice.furnsh(spice_kernel_file)  # spacecraft trajectory kernel

    def _spice_kernel_ephemeris(self) -> bool:
        """Calculate ephemeris based on SPICE kernels."""
        # Load SPICE kernels
        self._load_spice_kernels()

        start_et = spice.str2et(str(self.begin.datetime))
        end_et = start_et + self.step_size.to_value(u.s) * len(self.timestamp)

        # Generate array of times (one-minute intervals)
        time_intervals = np.arange(start_et, end_et, 60)  # 60s = 1 min

        # Compute full state vector (position + velocity) in batch for J2000 (GCRS)
        states = np.array(
            [spice.spkezr(str(self.naif_id), et, "J2000", "NONE", "399")[0] for et in time_intervals]
        )

        # Extract position and velocity from state vectors
        positions_gcrs = states[:, :3]  # First three elements (X, Y, Z) [km]
        velocities_gcrs = states[:, 3:]  # Last three elements (Vx, Vy, Vz) [km/s]

        # Create GCRS coordinates
        gcrs_p = CartesianRepresentation(positions_gcrs.T * u.km)
        gcrs_v = CartesianDifferential(velocities_gcrs.T * u.km / u.s)
        self.gcrs = SkyCoord(gcrs_p.with_differentials(gcrs_v), frame=GCRS(obstime=self.timestamp))

        # Transform to ITRS and get Earth location
        itrs = self.gcrs.transform_to("itrs")
        self.earth_location = itrs.earth_location

        return True

    def _jpl_horizons_ephemeris(self) -> bool:
        """Calculate ephemeris based on JPL Horizons data."""
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

        return True

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

    def _ephemeris_calc(self) -> bool:
        """Calculate ephemeris data based on the computed coordinates."""
        # Calculate the position of the Moon relative to the spacecraft
        self.moon = get_body("moon", self.timestamp, location=self.earth_location)

        # Calculate the position of the Sun relative to the spacecraft
        self.sun = get_body("sun", self.timestamp, location=self.earth_location)

        # Calculate the position of the Earth relative to the spacecraft
        self.earth = get_body("earth", self.timestamp, location=self.earth_location)

        # Calculate the latitude, longitude and distance from the center of the
        # Earth of the satellite
        self.longitude = self.earth_location.lon
        self.latitude = self.earth_location.lat
        self.height = self.earth_location.height
        self.distance = self.gcrs.distance

        # Calculate the Earth angular radius as seen from the spacecraft
        self.earth_radius_angle = np.arcsin(R_earth / self.distance)

        # Similarly calculate the angular radii of the Sun and the Moon
        self.moon_radius_angle = np.arcsin(R_moon / self.moon.distance)
        self.sun_radius_angle = np.arcsin(R_sun / self.sun.distance)

        return True

    def compute(self, ephemeris_type: EphemerisType) -> bool:
        """
        Compute ephemeris based on the specified type.
        This method calculates the ephemeris for an astronomical object based on the provided
        ephemeris type, handling ground-based observations, space-based TLE calculations,
        and JPL Horizons ephemeris.
        Parameters
        ----------
        ephemeris_type : EphemerisType
            The type of ephemeris calculation to perform. Must be one of:
            - EphemerisType.GROUND_BASED
            - EphemerisType.SPACE_TLE
            - EphemerisType.SPACE_JPL
            - EphemerisType.SPACE_SPICE

        Returns
        -------
        bool
            True if computation is successful

        Raises
        ------
        Exception
            If an invalid ephemeris type is provided (status_code=400)

        Notes
        -----
        The method performs the following steps:
        1. Calculates GCRS and EarthLocation coordinates based on ephemeris type.
        2. Computes final ephemeris calculations through _ephemeris_calc()
        """
        # Compute timestamps
        self.timestamp = self._compute_timestamp()

        # Calculate GCRS/ITRS and EarthLocation coordinates of Observatory
        # based on type of Ephemeris being calculated
        if ephemeris_type == EphemerisType.GROUND_BASED:
            self._ground_ephemeris()
        elif ephemeris_type == EphemerisType.SPACE_TLE:
            self._tle_ephemeris()
        elif ephemeris_type == EphemerisType.SPACE_JPL:
            self._jpl_horizons_ephemeris()
        elif ephemeris_type == EphemerisType.SPACE_SPICE:
            self._spice_kernel_ephemeris()
        else:
            raise Exception("Invalid ephemeris type")

        self._ephemeris_calc()
        return True


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

    ephemeris = Ephemeris(
        begin=begin, end=end, step_size=step_size, latitude=latitude, longitude=longitude, height=height
    )
    ephemeris.compute(ephemeris_type=EphemerisType.GROUND_BASED)
    return ephemeris


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
    ephemeris = Ephemeris(tle=tle, begin=begin, end=end, step_size=step_size)
    ephemeris.compute(ephemeris_type=EphemerisType.SPACE_TLE)
    return ephemeris


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
    ephemeris = Ephemeris(naif_id=naif_id, begin=begin, end=end, step_size=step_size)
    ephemeris.compute(ephemeris_type=EphemerisType.SPACE_JPL)
    return ephemeris


def compute_spice_ephemeris(
    begin: Union[datetime, Time],
    end: Union[datetime, Time],
    step_size: Union[int, timedelta, TimeDelta],
    spice_kernel_url: str,
    naif_id: int,
) -> Ephemeris:
    """
    Compute space ephemeris data using SPICE kernels.

    Parameters
    ----------
    begin : Union[datetime, Time]
        Start date and time for ephemeris computation
    end : Union[datetime, Time]
        End date and time for ephemeris computation
    step_size : Union[int, timedelta, TimeDelta]
        Time step size between ephemeris points, in seconds if int
    spice_kernel_url : str
        URL to the SPICE kernel file
    naif_id : int
        NAIF object identifier (e.g., 301 for Moon, -48 for HST)

    Returns
    -------
    Ephemeris
        An Ephemeris object containing the computed ephemeris data

    Notes
    -----
    This function uses the SPICE system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephemeris = compute_spice_ephemeris(begin, end, 60, 'https://path/to/spice_kernel.bsp', 301)
    """
    ephemeris = Ephemeris(
        naif_id=naif_id, begin=begin, end=end, step_size=step_size, spice_kernel_url=spice_kernel_url
    )
    ephemeris.compute(ephemeris_type=EphemerisType.SPACE_SPICE)
    return ephemeris
