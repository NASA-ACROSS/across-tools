Ephemeris Generation
========================================================================================

The ``across.tools.ephemeris`` module provides classes and functions for computing 
ephemerides of astronomical observatories, both ground-based and space-based. An 
ephemeris describes the position of an observatory over time, which is essential for 
visibility calculations and observational planning.

Overview
--------

The ephemeris module supports four different sources of ephemeris data:

- **TLE (Two-Line Element)**: For Earth-orbiting satellites using SGP4 propagation
- **JPL Horizons**: For spacecraft and celestial bodies using NASA's JPL Horizons system
- **SPICE**: For high-precision spacecraft trajectories using NASA NAIF SPICE kernels
- **Ground**: For fixed ground-based observatories

All ephemeris classes inherit from a common ``Ephemeris`` base class and provide 
consistent interfaces for accessing computed positions, celestial body locations, 
and angular separations.

Base Ephemeris Class
--------------------

All ephemeris types share common attributes computed by the base ``Ephemeris`` class:

.. list-table:: Common Ephemeris Attributes
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Description
   * - ``timestamp``
     - Array of calculation timestamps (``astropy.time.Time``)
   * - ``gcrs``
     - Observatory position in Geocentric Celestial Reference System (``SkyCoord``)
   * - ``earth_location``
     - Observatory position as ``EarthLocation``
   * - ``sun``
     - Sun position relative to observatory (``SkyCoord``)
   * - ``moon``
     - Moon position relative to observatory (``SkyCoord``)
   * - ``earth``
     - Earth position relative to observatory (``SkyCoord``)
   * - ``longitude``
     - Observatory longitude
   * - ``latitude``
     - Observatory latitude
   * - ``height``
     - Observatory height above Earth's surface
   * - ``distance``
     - Distance from observatory to Earth center
   * - ``earth_radius_angle``
     - Angular radius of Earth as seen from observatory
   * - ``moon_radius_angle``
     - Angular radius of Moon as seen from observatory
   * - ``sun_radius_angle``
     - Angular radius of Sun as seen from observatory

TLE Ephemeris
-------------

The ``TLEEphemeris`` class computes satellite positions using Two-Line Element (TLE) 
data and the SGP4 propagator. This is the most common method for tracking Earth-orbiting 
satellites.

**When to use**: Low Earth Orbit (LEO) satellites, ISS, Hubble Space Telescope, Swift, etc.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_tle_ephemeris
   from across.tools.tle import get_tle

   # Fetch TLE data from Space-Track.org (requires credentials)
   tle = get_tle(
       norad_id=28485,  # Swift satellite
       epoch=datetime(2024, 1, 1),
       spacetrack_user="your_username",
       spacetrack_pwd="your_password"
   )

   # Compute ephemeris
   ephem = compute_tle_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 2),
       step_size=60,  # seconds
       tle=tle
   )

   # Access computed data
   print(f"Number of time steps: {len(ephem.timestamp)}")
   print(f"Satellite altitude: {ephem.height}")

Using TLE Strings Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have TLE strings, you can create a TLE object directly:

.. code-block:: python

   from across.tools.core.schemas.tle import TLE

   tle = TLE(
       norad_id=25544,
       satellite_name="ISS (ZARYA)",
       tle1="1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9025",
       tle2="2 25544  51.6400 208.9163 0006703 130.5360 325.0288 15.49560532    18"
   )

Parameters
^^^^^^^^^^

.. list-table:: TLEEphemeris Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``begin``
     - datetime/Time
     - Start time of ephemeris calculation
   * - ``end``
     - datetime/Time
     - End time of ephemeris calculation
   * - ``step_size``
     - int/TimeDelta
     - Time step in seconds (default: 60)
   * - ``tle``
     - TLE
     - TLE data object containing orbital elements

Fetching TLE Data
^^^^^^^^^^^^^^^^^

The ``get_tle`` function fetches TLE data from Space-Track.org:

.. code-block:: python

   from across.tools.tle import get_tle

   # Credentials can be provided directly or via environment variables
   # SPACETRACK_USER and SPACETRACK_PWD
   tle = get_tle(
       norad_id=28485,
       epoch=datetime(2024, 1, 1),
       spacetrack_user="your_username",  # optional if env vars set
       spacetrack_pwd="your_password"    # optional if env vars set
   )

Common NORAD IDs:

- ISS (ZARYA): 25544
- Hubble Space Telescope: 20580
- Swift: 28485
- Fermi: 33053
- IXPE: 49954

JPL Horizons Ephemeris
----------------------

The ``JPLEphemeris`` class retrieves ephemeris data from NASA's JPL Horizons system. 
This is useful for spacecraft with ephemerides available in Horizons or for celestial 
bodies.

**When to use**: Planetary missions, deep space missions, or when high-precision 
ephemerides are available in JPL Horizons.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_jpl_ephemeris

   # Compute ephemeris for Hubble Space Telescope (NAIF ID: -48)
   ephem = compute_jpl_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 2),
       step_size=60,  # seconds
       naif_id=-48    # HST
   )

   print(f"HST position: {ephem.gcrs}")

Parameters
^^^^^^^^^^

.. list-table:: JPLEphemeris Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``begin``
     - datetime/Time
     - Start time of ephemeris calculation
   * - ``end``
     - datetime/Time
     - End time of ephemeris calculation
   * - ``step_size``
     - int/TimeDelta
     - Time step in seconds (default: 60)
   * - ``naif_id``
     - int
     - NAIF ID of the spacecraft or celestial body

Common NAIF IDs:

- Moon: 301
- Sun: 10
- Hubble Space Telescope: -48
- James Webb Space Telescope: -170
- Chandra X-ray Observatory: -151

SPICE Ephemeris
---------------

The ``SPICEEphemeris`` class computes high-precision ephemerides using NASA NAIF SPICE 
kernels. This provides the most accurate positions when SPICE kernels are available.

**When to use**: When high-precision ephemerides are required and SPICE kernels are 
available for the spacecraft.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_spice_ephemeris

   # SPICE kernel URL for the spacecraft
   kernel_url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/.../spk_file.bsp"

   ephem = compute_spice_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 2),
       step_size=60,
       spice_kernel_url=kernel_url,
       naif_id=-170  # JWST
   )

Parameters
^^^^^^^^^^

.. list-table:: SPICEEphemeris Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``begin``
     - datetime/Time
     - Start time of ephemeris calculation
   * - ``end``
     - datetime/Time
     - End time of ephemeris calculation
   * - ``step_size``
     - int/TimeDelta
     - Time step in seconds (default: 60)
   * - ``spice_kernel_url``
     - str
     - URL to download the spacecraft SPICE kernel
   * - ``naif_id``
     - int
     - NAIF ID of the spacecraft

Required SPICE Kernels
^^^^^^^^^^^^^^^^^^^^^^

The ``SPICEEphemeris`` class automatically downloads and loads the following 
standard kernels:

- Leap seconds kernel (naif0012.tls)
- Planetary ephemeris (de442s.bsp)
- Earth orientation parameters (earth_latest_high_prec.bpc)
- User-provided spacecraft kernel

Ground Ephemeris
----------------

The ``GroundEphemeris`` class computes ephemeris data for fixed ground-based 
observatories. While the observatory doesn't move, this class computes the 
positions of celestial bodies (Sun, Moon) relative to the observatory location.

**When to use**: Ground-based telescopes and observatories.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   import astropy.units as u
   from astropy.coordinates import Latitude, Longitude
   from across.tools.ephemeris import compute_ground_ephemeris

   # Compute ephemeris for an observatory
   ephem = compute_ground_ephemeris(
       begin=datetime(2024, 6, 21),
       end=datetime(2024, 6, 22),
       step_size=3600,  # 1 hour
       latitude=Latitude(34.0 * u.deg),
       longitude=Longitude(-118.0 * u.deg),
       height=100.0 * u.m
   )

   print(f"Sun position: {ephem.sun}")
   print(f"Moon position: {ephem.moon}")

Parameters
^^^^^^^^^^

.. list-table:: GroundEphemeris Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``begin``
     - datetime/Time
     - Start time of ephemeris calculation
   * - ``end``
     - datetime/Time
     - End time of ephemeris calculation
   * - ``step_size``
     - int/TimeDelta
     - Time step in seconds (default: 60)
   * - ``latitude``
     - Latitude
     - Observatory latitude
   * - ``longitude``
     - Longitude
     - Observatory longitude
   * - ``height``
     - Quantity
     - Observatory height above sea level

Working with Ephemeris Data
---------------------------

Time Indexing
^^^^^^^^^^^^^

You can find the index for a specific time in the ephemeris:

.. code-block:: python

   from astropy.time import Time

   # Find index for a specific time
   t = Time("2024-01-01T12:00:00")
   idx = ephem.index(t)

   # Access data at that time
   print(f"Position at {t}: {ephem.gcrs[idx]}")

Accessing Celestial Body Positions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Sun and Moon positions are available for all ephemeris types
   sun_positions = ephem.sun
   moon_positions = ephem.moon

   # Angular sizes of celestial bodies
   earth_angular_radius = ephem.earth_radius_angle
   sun_angular_radius = ephem.sun_radius_angle
   moon_angular_radius = ephem.moon_radius_angle

Coordinate Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``gcrs`` attribute provides positions in the Geocentric Celestial Reference 
System, which can be transformed to other coordinate systems using astropy:

.. code-block:: python

   # Transform to other coordinate systems
   icrs = ephem.gcrs.transform_to('icrs')
   galactic = ephem.gcrs.transform_to('galactic')

Choosing the Right Ephemeris Type
---------------------------------

.. list-table:: Ephemeris Type Selection Guide
   :widths: 25 25 50
   :header-rows: 1

   * - Observatory Type
     - Recommended Class
     - Notes
   * - LEO Satellites
     - ``TLEEphemeris``
     - Most common; uses freely available TLE data
   * - Deep Space Missions
     - ``JPLEphemeris`` or ``SPICEEphemeris``
     - Use SPICE for highest precision
   * - Well-known Spacecraft
     - ``JPLEphemeris``
     - Easy access via JPL Horizons
   * - Ground Observatories
     - ``GroundEphemeris``
     - Fixed position; computes celestial body positions

API Reference
-------------

See the :doc:`API Reference </autoapi/across/tools/ephemeris/index>` for complete 
class and function documentation.
