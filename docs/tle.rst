TLE Data
========================================================================================

The ``across.tools.tle`` module provides functionality for fetching and working with 
Two-Line Element (TLE) data for Earth-orbiting satellites. TLEs are the standard format 
for describing satellite orbits and are essential for computing spacecraft ephemerides.

Overview
--------

Two-Line Element sets (TLEs) are a data format encoding orbital elements for 
Earth-orbiting objects. They are used with the SGP4 propagator to predict satellite 
positions. The TLE module provides:

- **TLE Fetching**: Retrieve TLE data from Space-Track.org
- **TLE Parsing**: Parse and validate TLE strings
- **Epoch Selection**: Automatically select the TLE closest to a requested epoch

Space-Track.org Account
-----------------------

To fetch TLE data, you need to create a free account at `Space-Track.org
<https://www.space-track.org/>`_, if you do not already have one.

**Setting Up Credentials**

You can provide credentials in three ways:

1. **Environment Variables** (recommended):

   .. code-block:: bash

      export SPACETRACK_USER="your_username"
      export SPACETRACK_PWD="your_password"

2. **Environment File** (``.env`` in your project directory):

   .. code-block:: text

      SPACETRACK_USER=your_username
      SPACETRACK_PWD=your_password

3. **Direct Parameters** (not recommended for production):

   .. code-block:: python

      tle = get_tle(
          norad_id=25544,
          epoch=datetime(2024, 1, 1),
          spacetrack_user="your_username",
          spacetrack_pwd="your_password"
      )

Fetching TLE Data
-----------------

Using get_tle()
^^^^^^^^^^^^^^^

The ``get_tle()`` function is the simplest way to fetch TLE data:

.. code-block:: python

   from datetime import datetime
   from across.tools.tle import get_tle

   # Fetch TLE for the ISS closest to a specific date
   tle = get_tle(
       norad_id=25544,           # ISS NORAD ID
       epoch=datetime(2024, 1, 1)
   )

   if tle is not None:
       print(f"Satellite: {tle.satellite_name}")
       print(f"NORAD ID: {tle.norad_id}")
       print(f"TLE Epoch: {tle.epoch}")
       print(f"Line 1: {tle.tle1}")
       print(f"Line 2: {tle.tle2}")
   else:
       print("No TLE found for the requested epoch")

The function searches for TLEs within ±7 days of the requested epoch and returns 
the one closest to the requested time.

.. list-table:: get_tle() Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``norad_id``
     - int
     - NORAD Catalog Number identifying the satellite
   * - ``epoch``
     - datetime
     - Target epoch for TLE selection
   * - ``spacetrack_user``
     - str (optional)
     - Space-Track.org username (uses env var if not provided)
   * - ``spacetrack_pwd``
     - str (optional)
     - Space-Track.org password (uses env var if not provided)

Using TLEFetch Class
^^^^^^^^^^^^^^^^^^^^

For more control, use the ``TLEFetch`` class directly:

.. code-block:: python

   from datetime import datetime
   from across.tools.tle import TLEFetch

   # Create a TLE fetcher
   fetcher = TLEFetch(
       norad_id=28485,                    # Swift satellite
       epoch=datetime(2024, 6, 15),
       satellite_name="SWIFT",            # Optional: set satellite name
   )

   # Fetch the TLE
   tle = fetcher.get()

TLE Data Structure
------------------

The ``TLE`` class represents a parsed Two-Line Element set:

.. code-block:: python

   from across.tools.core.schemas.tle import TLE

   # Create TLE from strings
   tle = TLE(
       norad_id=25544,
       satellite_name="ISS (ZARYA)",
       tle1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
       tle2="2 25544  51.6400 208.9163 0006703 130.5360 325.0288 15.49560532434567"
   )

   # The epoch is automatically parsed from the TLE
   print(f"Epoch: {tle.epoch}")

.. list-table:: TLE Attributes
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Description
   * - ``norad_id``
     - NORAD Catalog Number (SATCAT ID)
   * - ``satellite_name``
     - Name or designation of the satellite
   * - ``tle1``
     - First line of the TLE (69 characters)
   * - ``tle2``
     - Second line of the TLE (69 characters)
   * - ``epoch``
     - Epoch datetime parsed from the TLE data

TLE Format Reference
--------------------

A TLE consists of two 69-character lines:

**Line 1 Format**:

.. code-block:: text

   1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN

- Columns 3-7: NORAD Catalog Number
- Columns 10-17: International Designator
- Columns 19-32: Epoch (year and day fraction)
- Columns 34-43: First derivative of mean motion
- Columns 45-52: Second derivative of mean motion
- Columns 54-61: BSTAR drag term

**Line 2 Format**:

.. code-block:: text

   2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN

- Columns 3-7: NORAD Catalog Number
- Columns 9-16: Inclination (degrees)
- Columns 18-25: Right Ascension of Ascending Node (degrees)
- Columns 27-33: Eccentricity (decimal point assumed)
- Columns 35-42: Argument of Perigee (degrees)
- Columns 44-51: Mean Anomaly (degrees)
- Columns 53-63: Mean Motion (revolutions/day)
- Columns 64-68: Revolution number at epoch

Common NORAD IDs
----------------

.. list-table:: Common Satellite NORAD IDs
   :widths: 40 20 40
   :header-rows: 1

   * - Satellite
     - NORAD ID
     - Notes
   * - ISS (ZARYA)
     - 25544
     - International Space Station
   * - Hubble Space Telescope
     - 20580
     - HST
   * - Swift
     - 28485
     - Neil Gehrels Swift Observatory
   * - Fermi
     - 33053
     - Fermi Gamma-ray Space Telescope
   * - IXPE
     - 49954
     - Imaging X-ray Polarimetry Explorer
   * - NuSTAR
     - 38358
     - Nuclear Spectroscopic Telescope Array
   * - NICER
     - 42930
     - On ISS (separate ID for ISS platform)
   * - Chandra
     - 25867
     - Chandra X-ray Observatory

Integration with Ephemeris
--------------------------

TLEs are typically used with the ephemeris module to compute satellite positions:

.. code-block:: python

   from datetime import datetime
   from across.tools.tle import get_tle
   from across.tools.ephemeris import compute_tle_ephemeris

   # Step 1: Fetch TLE
   tle = get_tle(norad_id=28485, epoch=datetime(2024, 1, 1))

   # Step 2: Compute ephemeris
   ephem = compute_tle_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       step_size=60,
       tle=tle
   )

   # Step 3: Use ephemeris for visibility calculations
   from across.tools.visibility import compute_ephemeris_visibility
   from across.tools.visibility.constraints import SunAngleConstraint

   visibility = compute_ephemeris_visibility(
       ra=180.0,
       dec=45.0,
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       ephemeris=ephem,
       constraints=[SunAngleConstraint(min_angle=45)],
       observatory_name="Swift"
   )

Error Handling
--------------

The TLE module raises specific exceptions for error conditions:

.. code-block:: python

   from across.tools.tle import get_tle
   from across.tools.tle.exceptions import SpaceTrackAuthenticationError

   try:
       tle = get_tle(norad_id=25544, epoch=datetime(2024, 1, 1))
   except SpaceTrackAuthenticationError:
       print("Failed to authenticate with Space-Track.org")
       print("Check your SPACETRACK_USER and SPACETRACK_PWD credentials")

**Common Issues**:

- ``SpaceTrackAuthenticationError``: Invalid or missing credentials
- Returns ``None``: No TLE found within ±7 days of requested epoch
- Invalid TLE format: TLE lines must be exactly 69 characters

Best Practices
--------------

1. **Use Environment Variables**: Store credentials in environment variables rather 
   than hardcoding them in scripts.

2. **Cache TLEs**: If making multiple calculations for the same satellite and time 
   period, fetch the TLE once and reuse it.

3. **Check for None**: Always check if ``get_tle()`` returns ``None`` before using 
   the result.

4. **TLE Age**: TLEs degrade in accuracy over time. For best results, use TLEs 
   close to your observation epoch. The module automatically selects the closest 
   available TLE within ±7 days.

5. **Rate Limiting**: Space-Track.org has rate limits. Avoid making excessive 
   requests in short time periods.

API Reference
-------------

See the :doc:`API Reference </autoapi/across/tools/tle/index>` for complete 
class and function documentation.
