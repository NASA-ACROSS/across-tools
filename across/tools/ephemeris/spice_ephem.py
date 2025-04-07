from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import spiceypy as spice  # type: ignore[import-untyped]
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from astropy.utils.data import download_file  # type: ignore[import-untyped]

from .base import Ephemeris

NAIF_LEAP_SECONDS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
NAIF_PLANETARY_EPHEMERIS_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp"
NAIF_EARTH_ORIENTATION_PARAMETERS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc"
)


class SPICEEphemeris(Ephemeris):
    """
    SPICEEphemeris class for calculating spacecraft ephemeris using SPICE
    (Spacecraft Planet Instrument C-matrix Events) kernels.

    This class extends the base Ephemeris class to provide specialized
    functionality for computing spacecraft trajectories using  kernels.

    Parameters
    ----------
    begin : datetime or Time
        Start time of the ephemeris calculation
    end : datetime or Time
        End time of the ephemeris calculation
    step_size : int or TimeDelta or timedelta, default=60
        Time step between ephemeris points in seconds
    spice_kernel_url : str, optional
        URL to download the spacecraft SPICE kernel
    naif_id : int, optional
        Navigation and Ancillary Information Facility (NAIF) ID of the
        spacecraft/object in JPL Horizons or SPICE kernel

    Attributes
    ----------
    naif_id : int
        NAIF ID of object for JPL Horizons or SPICE Kernel
    spice_kernel_url : str
        URL of spacecraft SPICE Kernel
    gcrs : SkyCoord
        Calculated positions in GCRS frame
    earth_location : EarthLocation
        Calculated positions in ITRS frame

    Methods
    -------
    prepare_data()
        Loads necessary SPICE kernels and calculates spacecraft trajectory

    Notes
    -----
    Requires SPICE kernels for:
    - Leap seconds
    - Planetary ephemeris
    - Earth orientation parameters
    - Spacecraft trajectory
    The class automatically handles downloading and caching of required SPICE
    kernels.
    """

    # NAIF ID of object for JPL Horizons or Spice Kernel
    naif_id: int | None = None

    # URL of spacecraft SPICE Kernel
    spice_kernel_url: str | None = None

    def __init__(
        self,
        begin: datetime | Time,
        end: datetime | Time,
        step_size: int | TimeDelta | timedelta = 60,
        spice_kernel_url: str | None = None,
        naif_id: int | None = None,
    ) -> None:
        super().__init__(begin, end, step_size)
        self.spice_kernel_url = spice_kernel_url
        self.naif_id = naif_id

    def prepare_data(self) -> None:
        """Loading SPICE kernels to calculate SPICE based ephemeris."""
        # Check that parameters needed for SPICE are set
        if self.spice_kernel_url is None:
            raise ValueError("No SPICE kernel URL provided")
        if self.naif_id is None:
            raise ValueError("No NAIF ID provided")

        # Download and load SPICE kernels if not already loaded
        kernel_urls = [
            NAIF_LEAP_SECONDS_URL,
            NAIF_PLANETARY_EPHEMERIS_URL,
            NAIF_EARTH_ORIENTATION_PARAMETERS_URL,
            self.spice_kernel_url,
        ]
        loaded_kernels = [str(spice.kdata(i, "all")[0]) for i in range(spice.ktotal("all"))]

        for url in kernel_urls:
            kernel_file = download_file(url, cache=True)
            if kernel_file not in loaded_kernels:
                spice.furnsh(kernel_file)

        # Generate array of times for ephemeris calculation in Ephemeris time (ET) format
        start_et = spice.str2et(str(self.begin.datetime))
        end_et = start_et + self.step_size.to_value(u.s) * len(self.timestamp)
        time_intervals = np.arange(start_et, end_et, self.step_size.to_value(u.s))

        # Get position and velocity vectors (states) in GCRS/J2000 frame
        states = np.array(
            [spice.spkezr(str(self.naif_id), et, "J2000", "NONE", "399")[0] for et in time_intervals]
        )

        # Create GCRS SkyCoord from states array
        self.gcrs = SkyCoord(
            CartesianRepresentation(states[:, :3].T * u.km).with_differentials(
                CartesianDifferential(states[:, 3:].T * u.km / u.s)
            ),
            frame=GCRS(obstime=self.timestamp),
        )

        # Transform to ITRS and get Earth location
        self.earth_location = self.gcrs.transform_to("itrs").earth_location


def compute_spice_ephemeris(
    begin: datetime | Time,
    end: datetime | Time,
    step_size: int | timedelta | TimeDelta,
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
        Navigation and Ancillary Information Facility (NAIF) object identifier
        (e.g., 301 for Moon, -48 for HST)

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
    # Compute the ephemeris using the SPICEEphemeris class
    ephemeris = SPICEEphemeris(
        naif_id=naif_id, begin=begin, end=end, step_size=step_size, spice_kernel_url=spice_kernel_url
    )
    ephemeris.compute()
    return ephemeris
