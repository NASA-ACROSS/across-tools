import os
from datetime import datetime, timedelta
from typing import Optional, Union

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
SPICE_KERNEL_CACHE_DIR = os.path.expanduser("~/.cache/across/spice")


# Define the radii of the M
class SPICEEphemeris(Ephemeris):
    """
    Ephemeris for space objects using SPICE kernels.
    """

    # NAIF ID of object for JPL Horizons or Spice Kernel
    naif_id: Optional[int] = None

    # URL of spacecraft SPICE Kernel
    spice_kernel_url: Optional[str] = None

    def __init__(
        self,
        begin: Union[datetime, Time],
        end: Union[datetime, Time],
        step_size: Union[int, TimeDelta, timedelta] = 60,
        spice_kernel_url: Optional[str] = None,
        naif_id: Optional[int] = None,
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
    # Compute the ephemeris using the SPICEEphemeris class
    ephemeris = SPICEEphemeris(
        naif_id=naif_id, begin=begin, end=end, step_size=step_size, spice_kernel_url=spice_kernel_url
    )
    ephemeris.compute()
    return ephemeris
