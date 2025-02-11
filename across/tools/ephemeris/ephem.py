import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import aiofiles
import astropy.units as u  # type: ignore[import-untyped]
import httpx
import numpy as np
import spiceypy as spice  # type: ignore[import-untyped]
from astropy.constants import R_earth, R_sun  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    TEME,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
    get_body,
)
from astropy.time import Time  # type: ignore[import-untyped]
from astroquery.jplhorizons import Horizons  # type: ignore[import-untyped]
from sgp4.api import Satrec  # type: ignore[import-untyped]

from ..core.enums import EphemType
from ..core.schemas.tle import TLE

# Define the radii of the Moon (as astropy doesn't)
R_moon = 1737.4 * u.km

NAIF_LEAP_SECONDS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
NAIF_PLANETARY_EPHEMERIS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp"
NAIF_EARTH_ORIENTATION_PARAMETERS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc"
)
SPICE_KERNEL_CACHE_DIR = os.path.expanduser("~/.cache/across/spice")


class Ephem:
    """
    A class for handling astronomical ephemeris calculations.

    This class provides functionality to compute and manage ephemeris data for
    ground-based observatories, satellites (using TLE data), and solar system
    bodies (using JPL Horizons). It can calculate positions, velocities, and
    angular sizes of celestial bodies.

    begin : datetime
        Start time for ephemeris calculations
    end : datetime
        End time for ephemeris calculations
    stepsize : int, optional
        Time step in seconds between ephemeris points (default: 60)
    earth_radius : float, optional
        Custom Earth radius value in km (default: None)
    earth_location : EarthLocation
        Location on Earth for ground-based calculations
    tle : TLE, optional
        Two-line element set for satellite calculations
    naif_id : int
        JPL Horizons ID for solar system body calculations

    Attributes itrs : SkyCoord
        Position in International Terrestrial Reference System
    gcrs : SkyCoord
        Position in Geocentric Celestial Reference System
    posvec : CartesianRepresentation
        Position vector in cartesian coordinates
    velvec : CartesianDifferential
        Velocity vector in cartesian coordinates
    moon : SkyCoord
        Moon position relative to observer
    sun : SkyCoord
        Sun position relative to observer
    earth : SkyCoord
        Earth position relative to observer
    longitude : Longitude
        Observer's longitude
    latitude : Latitude
        Observer's latitude
    earthsize : u.Quantity
        Angular size of Earth as seen from observer
    moon_size : u.Quantity
        Angular size of Moon as seen from observer
    sun_size : u.Quantity
        Angular size of Sun as seen from observer

    Methods compute(ephem_type: EphemType)
        Compute ephemeris data synchronously
    compute_async(ephem_type: EphemType)
        Compute ephemeris data asynchronously
    ephindex(t: Time)
        Get index of nearest time in ephemeris
    timestamp
        Get array of timestamps based on time interval and step size
    _ephem_calc
        Calculate final ephemeris data
    _ground_ephem
        Calculate ground-based observatory coordinates
    _tle_ephem
        Calculate satellite position using TLE data
    _spice_kernel_ephem
        Calculate satellite position using SPICE kernel
    _jpl_horizons_ephem
        Calculate solar system body position using JPL Horizons

    Notes
    -----
    The class supports three types of ephemeris calculations: - Ground-based
    observations - Space-based observations using TLE data - Solar system body
    observations using JPL Horizons

    See Also
    --------
    astropy.coordinates.SkyCoord astropy.coordinates.EarthLocation
    """

    # Type hints
    begin: datetime
    end: datetime
    stepsize: int = 60

    timestamp: Time
    itrs: SkyCoord
    gcrs: SkyCoord
    posvec: CartesianRepresentation
    velvec: CartesianDifferential
    moon: SkyCoord
    sun: SkyCoord
    earth: SkyCoord
    longitude: Longitude
    latitude: Latitude
    height: u.Quantity
    earth_size: u.Quantity
    moon_size: u.Quantity
    sun_size: u.Quantity
    # Ephemeris attributes
    earth_location: EarthLocation
    tle: Optional[TLE]
    naif_id: int
    spice_kernel_url: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __len__(self) -> int:
        if self.timestamp is None:
            return 0
        return len(self.timestamp)

    def ephindex(self, t: Time) -> int:
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
        index = int(np.round((t.jd - self.timestamp[0].jd) // (self.stepsize / 86400)))
        assert index >= 0 and index < len(self), "Time outside of ephemeris of range"
        return index

    def _ground_ephem(self) -> bool:
        # Check if EarthLocation is set, if not, set it based on latitude, longitude, and height.
        if self.earth_location is None:
            if self.latitude is None or self.longitude is None or self.height is None:
                raise Exception("Location of observatory not set")
            else:
                self.earth_location = EarthLocation.from_geodetic(self.latitude, self.longitude, self.height)

        # Calculate GCRS and ITRS coordinates of the observatory
        self.gcrs = self.earth_location.get_gcrs(self.timestamp)
        self.itrs = self.earth_location.get_itrs(self.timestamp)
        return True

    def _tle_ephem(self) -> bool:
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
        self.itrs = SkyCoord(teme_p.with_differentials(teme_v), frame=TEME(obstime=self.timestamp)).itrs
        self.earth_location = self.itrs.earth_location

        # Calculate satellite position in GCRS coordinate system vector as
        # array of x,y,z vectors in units of km, and velocity vector as array
        # of x,y,z vectors in units of km/s
        self.gcrs = self.itrs.transform_to(GCRS)
        return True

    def _load_spice_kernels(self) -> None:
        """Synchronous version of loading spice kernels"""
        if self.spice_kernel_url is None:
            raise Exception("No SPICE kernel URL provided")

        # Create cache directory
        os.makedirs(SPICE_KERNEL_CACHE_DIR, exist_ok=True)

        # Download all required kernels in one pass
        urls = [
            NAIF_LEAP_SECONDS_URL,
            NAIF_PLANETARY_EPHEMERIS_URL,
            NAIF_EARTH_ORIENTATION_PARAMETERS_URL,
            self.spice_kernel_url,
        ]

        with httpx.Client() as client:
            for url in urls:
                local_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(url))
                if not os.path.exists(local_file):
                    response = client.get(url)
                    response.raise_for_status()
                    with open(local_file, "wb") as f:
                        f.write(response.content)

        self._load_kernel_files()

    async def _download_file(self, client: httpx.AsyncClient, url: str, local_file: str) -> None:
        response = await client.get(url)
        response.raise_for_status()
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(response.content)

    async def _load_spice_kernels_async(self) -> None:
        # Async version of loading spice kernels
        if self.spice_kernel_url is None:
            raise Exception("No SPICE kernel URL provided")

        # Create cache directory if it doesn't exist
        os.makedirs(SPICE_KERNEL_CACHE_DIR, exist_ok=True)

        # Download all required kernels in one pass
        urls = [
            NAIF_LEAP_SECONDS_URL,
            NAIF_PLANETARY_EPHEMERIS_URL,
            NAIF_EARTH_ORIENTATION_PARAMETERS_URL,
            self.spice_kernel_url,
        ]

        # Download spice kernels if they don't exist locally
        async with httpx.AsyncClient() as client:
            downloads = []
            for url in urls:
                local_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(url))

                if not os.path.exists(local_file):
                    downloads.append(self._download_file(client, url, local_file))
            if downloads:
                await asyncio.gather(*downloads)

        await self._load_kernel_files_async()

    def _load_kernel_files(self) -> None:
        if self.spice_kernel_url is None:
            raise Exception("No SPICE kernel URL provided")
        # Helper method to load the kernel files after download

        leap_seconds_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_LEAP_SECONDS_URL))
        planetary_ephem_file = os.path.join(
            SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_PLANETARY_EPHEMERIS_URL)
        )
        earth_params_file = os.path.join(
            SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_EARTH_ORIENTATION_PARAMETERS_URL)
        )
        spice_kernel_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(self.spice_kernel_url))

        # Check if kernels are already loaded
        loaded_kernels = [str(spice.kdata(i, "all")[0]) for i in range(spice.ktotal("all"))]

        # Load local cached kernel files if not already loaded
        if leap_seconds_file not in loaded_kernels:
            spice.furnsh(leap_seconds_file)  # Leap seconds
        if planetary_ephem_file not in loaded_kernels:
            spice.furnsh(planetary_ephem_file)  # Planetary ephemeris
        if earth_params_file not in loaded_kernels:
            spice.furnsh(earth_params_file)  # High-precision Earth orientation
        if spice_kernel_file not in loaded_kernels:
            spice.furnsh(spice_kernel_file)  # spacecraft trajectory kernel

    async def _load_kernel_files_async(self) -> None:
        if self.spice_kernel_url is None:
            raise Exception("No SPICE kernel URL provided")

        leap_seconds_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_LEAP_SECONDS_URL))
        planetary_ephem_file = os.path.join(
            SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_PLANETARY_EPHEMERIS_URL)
        )
        earth_params_file = os.path.join(
            SPICE_KERNEL_CACHE_DIR, os.path.basename(NAIF_EARTH_ORIENTATION_PARAMETERS_URL)
        )
        spice_kernel_file = os.path.join(SPICE_KERNEL_CACHE_DIR, os.path.basename(self.spice_kernel_url))

        # Check if kernels are already loaded
        loaded_kernels = [str(spice.kdata(i, "all")[0]) for i in range(spice.ktotal("all"))]

        # Load local cached kernel files if not already loaded
        if leap_seconds_file not in loaded_kernels:
            await asyncio.to_thread(spice.furnsh, leap_seconds_file)  # Leap seconds
        if planetary_ephem_file not in loaded_kernels:
            await asyncio.to_thread(spice.furnsh, planetary_ephem_file)  # Planetary ephemeris
        if earth_params_file not in loaded_kernels:
            await asyncio.to_thread(spice.furnsh, earth_params_file)  # High-precision Earth orientation
        if spice_kernel_file not in loaded_kernels:
            await asyncio.to_thread(spice.furnsh, spice_kernel_file)  # spacecraft trajectory kernel

    def _spice_kernel_ephem(self) -> bool:
        # Load SPICE kernels
        self._load_spice_kernels()

        start_et = spice.str2et(str(self.begin))
        end_et = spice.str2et(str(self.end))

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
        self.itrs = self.gcrs.transform_to("itrs")
        self.earth_location = self.itrs.earth_location

        return True

    async def _spice_kernel_ephem_async(self) -> bool:
        # Load SPICE kernels
        await self._load_spice_kernels_async()

        start_et = spice.str2et(str(self.begin))
        end_et = spice.str2et(str(self.end))

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
        self.itrs = self.gcrs.transform_to("itrs")
        self.earth_location = self.itrs.earth_location

        return True

    def _jpl_horizons_ephem(self) -> bool:
        # Create a time range dictionary for Horizons
        horizons_range = {
            "start": str(Time(self.begin).tdb),
            "stop": str(Time(self.end).tdb),
            "step": f"{self.stepsize / 60:.0f}m",
        }

        # Fetch the ephemeris vector data from Horizons
        horizons_ephem = Horizons(
            id=self.naif_id,
            location="500@399",
            epochs=horizons_range,
            id_type=None,
        )
        horizons_vectors = horizons_ephem.vectors(refplane="earth")

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
        self.itrs = self.gcrs.itrs
        self.earth_location = self.itrs.earth_location

        return True

    async def _jpl_horizons_ephem_async(self) -> bool:
        # Create a time range dictionary for Horizons
        horizons_range = {
            "start": str(Time(self.begin).tdb),
            "stop": str(Time(self.end).tdb),
            "step": f"{self.stepsize / 60:.0f}m",
        }

        # Fetch the ephemeris vector data from Horizons
        horizons_ephem = Horizons(
            id=self.naif_id,
            location="500@399",
            epochs=horizons_range,
            id_type=None,
        )
        horizons_vectors = await asyncio.to_thread(horizons_ephem.vectors, refplane="earth")

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
        self.itrs = self.gcrs.itrs
        self.earth_location = self.itrs.earth_location

        return True

    def _compute_timestamp(self) -> Time:
        """
        Get array of timestamps based on time interval and step size.
        Returns
        -------
        astropy.time.Time
            If begin equals end, returns single timestamp.
            Otherwise returns array of timestamps from begin to end with specified stepsize.
        """

        # Create array of timestamps
        if self.begin == self.end:
            return Time([self.begin])
        step = timedelta(seconds=self.stepsize)
        return Time(np.arange(self.begin, self.end + step, step))

    def _ephem_calc(self, ephem_type: EphemType = EphemType.space_tle) -> bool:
        # Calculate the postion and velocity vectors
        self.posvec = self.gcrs.cartesian.without_differentials()
        self.velvec = self.gcrs.velocity.to_cartesian()

        # Calculate the position of the Moon relative to the spacecraft
        self.moon = get_body("moon", self.timestamp, location=self.earth_location)

        # Calculate the position of the Moon relative to the spacecraft
        self.sun = get_body("sun", self.timestamp, location=self.earth_location)

        # Calculate the position of the Earth relative to the spacecraft
        self.earth = get_body("earth", self.timestamp, location=self.earth_location)

        # Calculate the latitude, longitude and distance from the center of the
        # Earth of the satellite
        self.longitude = self.earth_location.lon
        self.latitude = self.earth_location.lat
        self.height = self.earth_location.height
        self.distance = self.posvec.norm()

        # Calculate the Earth radius in degrees
        self.earthsize = np.arcsin(R_earth / self.distance)

        # Similarly calculate the angular radii of the Sun and the Moon
        self.moon_size = np.arcsin(R_moon / self.moon.distance)
        self.sun_size = np.arcsin(R_sun / self.sun.distance)

        return True

    def compute(self, ephem_type: EphemType) -> bool:
        """
        Compute ephemeris based on the specified type.
        This method calculates the ephemeris for an astronomical object based on the provided
        ephemeris type, handling ground-based observations, space-based TLE calculations,
        and JPL Horizons ephemeris.
        Parameters
        ----------
        ephem_type : EphemType
            The type of ephemeris calculation to perform. Must be one of:
            - EphemType.ground_based
            - EphemType.space_tle
            - EphemType.space_jpl
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
        1. Calculates GCRS/ITRS coordinates based on ephemeris type
        2. Computes final ephemeris calculations through _ephem_calc()
        """
        # Compute timestamps
        self.timestamp = self._compute_timestamp()

        # Calculate GCRS/ITRS and EarthLocation coordinates of Observatory
        # based on type of Ephemeris being calculated
        if ephem_type == EphemType.ground_based:
            self._ground_ephem()
        elif ephem_type == EphemType.space_tle:
            self._tle_ephem()
        elif ephem_type == EphemType.space_jpl:
            self._jpl_horizons_ephem()
        elif ephem_type == EphemType.space_spice:
            self._spice_kernel_ephem()
        else:
            raise Exception("Invalid ephemeris type")

        self._ephem_calc()
        return True

    async def compute_async(self, ephem_type: EphemType) -> bool:
        """
        Asynchronously compute ephemeris based on the specified type.
        Parameters
        ----------
        ephem_type : EphemType
            The type of ephemeris calculation to perform (ground_based, space_tle, or space_jpl)
        Returns
        -------
        bool
            True if computation is successful
        Raises
        ------
        Exception
            If an invalid ephemeris type is provided (400 status code)
        Notes
        -----
        This method performs different ephemeris calculations based on the type:
        - For ground_based: Calculates ground-based observatory coordinates
        - For space_tle: Computes satellite positions using TLE data
        - For space_jpl: Gets positions using JPL HORIZONS system asynchronously
        The results are stored in instance variables and final calculations are
        performed via _ephem_calc() after the initial computations.
        """
        # Compute timestamps
        self.timestamp = self._compute_timestamp()

        # Calculate GCRS/ITRS and EarthLocation coordinates of Observatory
        # based on type of Ephemeris being calculated
        if ephem_type == EphemType.ground_based:
            self._ground_ephem()
        elif ephem_type == EphemType.space_tle:
            self._tle_ephem()
        elif ephem_type == EphemType.space_jpl:
            await self._jpl_horizons_ephem_async()
        elif ephem_type == EphemType.space_spice:
            await self._spice_kernel_ephem_async()
        else:
            raise Exception("Invalid ephemeris type")
        self._ephem_calc()
        return True


def compute_ground_ephem(
    begin: datetime,
    end: datetime,
    stepsize: int,
    latitude: Latitude,
    longitude: Longitude,
    height: u.Quantity,
) -> Ephem:
    """
    Compute ground-based ephemeris for a given time range and location.

    Parameters
    ----------
    begin : datetime
        The start time of the ephemeris computation.
    end : datetime
        The end time of the ephemeris computation.
    stepsize : int
        The step size in seconds for the ephemeris computation.
    latitude : Latitude
        The latitude of the ground-based observatory.
    longitude : Longitude
        The longitude of the ground-based observatory.
    height : u.Quantity
        The height of the ground-based observatory above sea level.

    Returns
    -------
    Ephem
        An Ephem object containing the computed ephemeris data.
    """

    ephem = Ephem(
        begin=begin, end=end, stepsize=stepsize, latitude=latitude, longitude=longitude, height=height
    )
    ephem.compute(ephem_type=EphemType.ground_based)
    return ephem


def compute_tle_ephem(begin: datetime, end: datetime, stepsize: int, tle: TLE) -> Ephem:
    """
    Compute the ephemeris for a space object using Two-Line Element (TLE) data.
    Parameters
    ----------
    begin : datetime
        The start time for the ephemeris computation.
    end : datetime
        The end time for the ephemeris computation.
    stepsize : int
        The time step size in seconds for the ephemeris computation.
    tle : TLE
        The TLE data entry for the space object.
    Returns
    -------
    Ephem
        The computed ephemeris object containing the position and velocity data.
    """
    ephem = Ephem(tle=tle, begin=begin, end=end, stepsize=stepsize)
    ephem.compute(ephem_type=EphemType.space_tle)
    return ephem


def compute_jpl_ephem(begin: datetime, end: datetime, stepsize: int, naif_id: int) -> Ephem:
    """
    Compute space ephemeris data using JPL Horizons system.

    Parameters
    ----------
    begin : datetime
        Start date and time for ephemeris computation
    end : datetime
        End date and time for ephemeris computation
    stepsize : int
        Time step size in seconds between ephemeris points
    naif_id : int
        NAIF object identifier (e.g., 301 for Moon. -48 for HST)

    Returns
    -------
    Ephem
        An Ephem object containing the computed ephemeris data

    Notes
    -----
    This function uses the JPL Horizons system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephem = compute_jpl_ephem(begin, end, 60, 301)
    """
    ephem = Ephem(naif_id=naif_id, begin=begin, end=end, stepsize=stepsize)
    ephem.compute(ephem_type=EphemType.space_jpl)
    return ephem


async def compute_jpl_ephem_async(begin: datetime, end: datetime, stepsize: int, naif_id: int) -> Ephem:
    """
    Compute space ephemeris data using JPL Horizons system. Async version (as
    JPL Horizons requires use of blocking IO).

    Parameters
    ----------
    begin : datetime
        Start date and time for ephemeris computation
    end : datetime
        End date and time for ephemeris computation
    stepsize : int
        Time step size in seconds between ephemeris points
    naif_id : int
        NAIF object identifier (e.g., 301 for Moon, -48 for HST)

    Returns
    -------
    Ephem
        An Ephem object containing the computed ephemeris data

    Notes
    -----
    This function uses the JPL Horizons system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephem = compute_jpl_ephem(begin, end, 60, 301)
    """
    ephem = Ephem(naif_id=naif_id, begin=begin, end=end, stepsize=stepsize)
    await ephem.compute_async(ephem_type=EphemType.space_jpl)
    return ephem


def compute_spice_ephem(
    begin: datetime, end: datetime, stepsize: int, spice_kernel_url: str, naif_id: int
) -> Ephem:
    """
    Compute space ephemeris data using SPICE kernels.

    Parameters
    ----------
    begin : datetime
        Start date and time for ephemeris computation
    end : datetime
        End date and time for ephemeris computation
    stepsize : int
        Time step size in seconds between ephemeris points
    spice_kernel_url : str
        URL to the SPICE kernel file
    naif_id : int
        NAIF object identifier (e.g., 301 for Moon, -48 for HST)

    Returns
    -------
    Ephem
        An Ephem object containing the computed ephemeris data

    Notes
    -----
    This function uses the SPICE system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephem = compute_spice_ephem(begin, end, 60, 'https://path/to/spice_kernel.bsp', 301)
    """
    ephem = Ephem(naif_id=naif_id, begin=begin, end=end, stepsize=stepsize, spice_kernel_url=spice_kernel_url)
    ephem.compute(ephem_type=EphemType.space_spice)
    return ephem


async def compute_spice_ephem_async(
    begin: datetime, end: datetime, stepsize: int, spice_kernel_url: str, naif_id: int
) -> Ephem:
    """
    Compute space ephemeris data using SPICE kernels. Async version (as
    SPICE requires use of blocking IO).

    Parameters
    ----------
    begin : datetime
        Start date and time for ephemeris computation
    end : datetime
        End date and time for ephemeris computation
    stepsize : int
        Time step size in seconds between ephemeris points
    spice_kernel_url : str
        URL to the SPICE kernel file
    naif_id : int
        NAIF object identifier (e.g., 301 for Moon, -48 for HST)

    Returns
    -------
    Ephem
        An Ephem object containing the computed ephemeris data

    Notes
    -----
    This function uses the SPICE system to compute high-precision
    ephemeris data for celestial bodies in space-based reference frame.
    Examples
    --------
    >>> from datetime import datetime
    >>> begin = datetime(2023, 1, 1)
    >>> end = datetime(2023, 1, 2)
    >>> moon_ephem = compute_spice_ephem(begin, end, 60, 'https://path/to/spice_kernel.bsp', 301)
    """
    ephem = Ephem(naif_id=naif_id, begin=begin, end=end, stepsize=stepsize, spice_kernel_url=spice_kernel_url)
    await ephem.compute_async(ephem_type=EphemType.space_spice)
    return ephem
