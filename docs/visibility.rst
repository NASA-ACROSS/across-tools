Visibility Calculation
========================================================================================

The ``across.tools.visibility`` module provides classes and functions for computing 
target visibility windows from astronomical observatories. It determines when celestial 
targets are observable by applying various observational constraints such as Sun angle, 
Moon angle, Earth limb avoidance, and more.

Overview
--------

The visibility module computes time windows during which a celestial target can be 
observed from a given observatory, accounting for:

- **Angular constraints**: Sun, Moon, and Earth limb avoidance
- **Geographic constraints**: South Atlantic Anomaly (SAA) avoidance
- **Local constraints**: Altitude and azimuth limits for ground observatories
- **Multi-observatory coordination**: Joint visibility across multiple instruments

The module provides two main visibility calculators:

- **EphemerisVisibility**: Computes visibility for a single observatory using ephemeris data
- **JointVisibility**: Computes combined visibility across multiple observatories

Visibility Constraints
----------------------

Constraints define conditions that must be met for a target to be observable. When a 
constraint is violated (returns ``True``), the target is considered *not* visible.

Sun Angle Constraint
^^^^^^^^^^^^^^^^^^^^

The ``SunAngleConstraint`` restricts observations based on the angular separation 
between the target and the Sun. This is essential for protecting instruments from 
direct sunlight and reducing scattered light.

.. code-block:: python

   from across.tools.visibility.constraints import SunAngleConstraint

   # Target must be at least 45° from the Sun
   sun_constraint = SunAngleConstraint(min_angle=45)

   # Target must be within 135° of the Sun (anti-Sun constraint)
   anti_sun = SunAngleConstraint(max_angle=135)

   # Combined: target between 45° and 135° from the Sun
   combined = SunAngleConstraint(min_angle=45, max_angle=135)

.. list-table:: SunAngleConstraint Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``min_angle``
     - float
     - Minimum angle from the Sun in degrees (0-180)
   * - ``max_angle``
     - float
     - Maximum angle from the Sun in degrees (0-180)

Moon Angle Constraint
^^^^^^^^^^^^^^^^^^^^^

The ``MoonAngleConstraint`` restricts observations based on the angular separation 
between the target and the Moon. This helps avoid scattered moonlight contamination.

.. code-block:: python

   from across.tools.visibility.constraints import MoonAngleConstraint

   # Target must be at least 15° from the Moon
   moon_constraint = MoonAngleConstraint(min_angle=15)

   # Target must be within 90° of the Moon
   near_moon = MoonAngleConstraint(max_angle=90)

.. list-table:: MoonAngleConstraint Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``min_angle``
     - float
     - Minimum angle from the Moon in degrees (0-180)
   * - ``max_angle``
     - float
     - Maximum angle from the Moon in degrees (0-180)

Earth Limb Constraint
^^^^^^^^^^^^^^^^^^^^^

The ``EarthLimbConstraint`` restricts observations for space-based observatories based 
on the angular distance from Earth's limb. This prevents Earth occultation and reduces 
atmospheric airglow contamination.

.. code-block:: python

   from across.tools.visibility.constraints import EarthLimbConstraint

   # Target must be at least 20° above Earth's limb
   earth_constraint = EarthLimbConstraint(min_angle=20)

.. list-table:: EarthLimbConstraint Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``min_angle``
     - float
     - Minimum angle above Earth's limb in degrees (0-180)
   * - ``max_angle``
     - float
     - Maximum angle above Earth's limb in degrees (0-180)

.. note::
   The Earth limb constraint accounts for the varying angular size of Earth as seen 
   from different orbital altitudes. The angular radius is computed dynamically from 
   the ephemeris data.

South Atlantic Anomaly (SAA) Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``SAAPolygonConstraint`` restricts observations when the spacecraft is within 
the South Atlantic Anomaly region, where increased radiation can affect instruments.

.. code-block:: python

   from shapely import Polygon
   from across.tools.visibility.constraints import SAAPolygonConstraint

   # Define SAA polygon (longitude, latitude coordinates)
   saa_polygon = Polygon([
       (-90, -30), (-40, -30), (-30, 0), (-40, 5),
       (-90, 5), (-90, -30)
   ])

   saa_constraint = SAAPolygonConstraint(polygon=saa_polygon)

.. list-table:: SAAPolygonConstraint Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``polygon``
     - Polygon
     - Shapely Polygon defining the SAA boundary in (longitude, latitude)

Altitude/Azimuth Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``AltAzConstraint`` restricts observations for ground-based observatories based 
on altitude and azimuth limits. This accounts for horizon obstructions, zenith blind 
spots, and other local constraints.

.. code-block:: python

   from across.tools.visibility.constraints import AltAzConstraint

   # Target must be above 30° altitude
   alt_constraint = AltAzConstraint(altitude_min=30)

   # Target must be between 30° and 85° altitude
   alt_range = AltAzConstraint(altitude_min=30, altitude_max=85)

   # Azimuth restriction (e.g., building obstruction)
   az_constraint = AltAzConstraint(azimuth_min=45, azimuth_max=135)

You can also define complex exclusion regions using a Shapely polygon:

.. code-block:: python

   from shapely import Polygon

   # Define an exclusion region in (altitude, azimuth) space
   exclusion = Polygon([(30, 0), (30, 45), (60, 45), (60, 0)])
   complex_constraint = AltAzConstraint(polygon=exclusion)

.. list-table:: AltAzConstraint Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``altitude_min``
     - float
     - Minimum altitude in degrees (0-90)
   * - ``altitude_max``
     - float
     - Maximum altitude in degrees (0-90)
   * - ``azimuth_min``
     - float
     - Minimum azimuth in degrees (0-360)
   * - ``azimuth_max``
     - float
     - Maximum azimuth in degrees (0-360)
   * - ``polygon``
     - Polygon
     - Shapely Polygon defining exclusion region in (alt, az) space

Ephemeris Visibility
--------------------

The ``EphemerisVisibility`` class computes visibility windows for a single observatory 
using pre-computed ephemeris data and a set of constraints.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_tle_ephemeris
   from across.tools.tle import get_tle
   from across.tools.visibility import compute_ephemeris_visibility
   from across.tools.visibility.constraints import (
       SunAngleConstraint,
       MoonAngleConstraint,
       EarthLimbConstraint,
   )

   # First, compute ephemeris for the spacecraft
   tle = get_tle(norad_id=28485, epoch=datetime(2024, 1, 1))
   ephem = compute_tle_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       step_size=60,
       tle=tle
   )

   # Define constraints
   constraints = [
       SunAngleConstraint(min_angle=45),
       MoonAngleConstraint(min_angle=15),
       EarthLimbConstraint(min_angle=20),
   ]

   # Compute visibility for a target
   visibility = compute_ephemeris_visibility(
       ra=180.0,              # Right ascension in degrees
       dec=45.0,              # Declination in degrees
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       ephemeris=ephem,
       constraints=constraints,
       observatory_name="Swift",
   )

   # Access visibility windows
   print(f"Found {len(visibility.visibility_windows)} visibility windows")
   for window in visibility.visibility_windows:
       print(f"  {window.window.begin.datetime} to {window.window.end.datetime}")
       print(f"    Duration: {window.max_visibility_duration} seconds")
       print(f"    Start reason: {window.constraint_reason.start_reason}")
       print(f"    End reason: {window.constraint_reason.end_reason}")

Using SkyCoord Instead of RA/Dec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also specify the target using an astropy ``SkyCoord`` object:

.. code-block:: python

   from astropy.coordinates import SkyCoord
   import astropy.units as u

   target = SkyCoord(ra=180*u.deg, dec=45*u.deg)

   visibility = compute_ephemeris_visibility(
       coordinate=target,
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       ephemeris=ephem,
       constraints=constraints,
       observatory_name="Swift",
   )

Parameters
^^^^^^^^^^

.. list-table:: compute_ephemeris_visibility Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``ra``
     - float
     - Right ascension in degrees (0-360)
   * - ``dec``
     - float
     - Declination in degrees (-90 to 90)
   * - ``coordinate``
     - SkyCoord
     - Alternative to ra/dec specification
   * - ``begin``
     - datetime/Time
     - Start time for visibility calculation
   * - ``end``
     - datetime/Time
     - End time for visibility calculation
   * - ``ephemeris``
     - Ephemeris
     - Pre-computed ephemeris for the observatory
   * - ``constraints``
     - list[Constraint]
     - List of constraint objects to apply
   * - ``step_size``
     - Quantity
     - Time step size (default: 60 seconds)
   * - ``observatory_name``
     - str
     - Name of the observatory
   * - ``observatory_id``
     - UUID
     - Unique identifier for the observatory (optional)
   * - ``min_vis``
     - int
     - Minimum visibility duration in seconds (default: 0)

Checking Visibility at Specific Times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After computing visibility, you can check if the target is visible at specific times:

.. code-block:: python

   from astropy.time import Time

   # Check single time
   t = Time("2024-01-01T12:00:00")
   is_visible = visibility.visible(t)
   print(f"Visible at {t}: {is_visible}")

   # Check multiple times
   times = Time(["2024-01-01T12:00:00", "2024-01-01T18:00:00", "2024-01-02T00:00:00"])
   visible_array = visibility.visible(times)
   print(f"Visibility: {visible_array}")

Joint Visibility
----------------

The ``JointVisibility`` class computes combined visibility windows across multiple 
observatories or instruments. This is useful for coordinated multi-observatory campaigns.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from uuid import uuid4
   from across.tools.visibility import compute_joint_visibility

   # Compute visibility for multiple observatories
   vis1 = compute_ephemeris_visibility(
       ra=180.0, dec=45.0,
       begin=begin, end=end,
       ephemeris=ephem1,
       constraints=constraints1,
       observatory_name="Observatory1",
   )

   vis2 = compute_ephemeris_visibility(
       ra=180.0, dec=45.0,
       begin=begin, end=end,
       ephemeris=ephem2,
       constraints=constraints2,
       observatory_name="Observatory2",
   )

   # Compute joint visibility (intersection of all visibility windows)
   joint = compute_joint_visibility(
       visibilities=[vis1, vis2],
       instrument_ids=[uuid4(), uuid4()],
   )

   print(f"Joint visibility windows: {len(joint.visibility_windows)}")
   for window in joint.visibility_windows:
       print(f"  {window.window.begin.datetime} to {window.window.end.datetime}")

Requirements for Joint Visibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All input visibility objects must have:

- The same target coordinates (within ~15 arcsec tolerance)
- The same begin and end times
- The same step size

Parameters
^^^^^^^^^^

.. list-table:: compute_joint_visibility Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``visibilities``
     - list[Visibility]
     - List of Visibility objects to combine
   * - ``instrument_ids``
     - list[UUID]
     - Unique identifiers for each instrument/observatory

Visibility Windows
------------------

Visibility calculations return ``VisibilityWindow`` objects containing detailed 
information about each visibility period.

VisibilityWindow Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VisibilityWindow Attributes
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Description
   * - ``window.begin``
     - Start of visibility window (ConstrainedDate with datetime and constraint info)
   * - ``window.end``
     - End of visibility window (ConstrainedDate with datetime and constraint info)
   * - ``max_visibility_duration``
     - Duration of window in seconds
   * - ``constraint_reason.start_reason``
     - Description of why visibility started (which constraint was satisfied)
   * - ``constraint_reason.end_reason``
     - Description of why visibility ended (which constraint was violated)

Example: Accessing Window Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   for window in visibility.visibility_windows:
       # Window timing
       start = window.window.begin.datetime
       end = window.window.end.datetime
       duration = window.max_visibility_duration

       # Constraint information
       start_constraint = window.window.begin.constraint
       end_constraint = window.window.end.constraint

       # Human-readable reasons
       start_reason = window.constraint_reason.start_reason
       end_reason = window.constraint_reason.end_reason

       print(f"Window: {start} to {end}")
       print(f"  Duration: {duration} seconds")
       print(f"  Started because: {start_reason}")
       print(f"  Ended because: {end_reason}")

Constraint Serialization
------------------------

Constraints can be serialized to/from JSON for storage or transmission:

.. code-block:: python

   from across.tools.visibility import constraints_to_json, constraints_from_json
   from across.tools.visibility.constraints import SunAngleConstraint, MoonAngleConstraint

   # Define constraints
   constraints = [
       SunAngleConstraint(min_angle=45, max_angle=180),
       MoonAngleConstraint(min_angle=15),
   ]

   # Serialize to JSON
   json_str = constraints_to_json(constraints)
   print(json_str)
   # Output: [{"name":"Sun Angle","short_name":"Sun","min_angle":45.0,"max_angle":180.0},...]

   # Deserialize from JSON
   loaded_constraints = constraints_from_json(json_str)

Complete Example
----------------

Here's a complete example showing a typical visibility calculation workflow:

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_tle_ephemeris
   from across.tools.tle import get_tle
   from across.tools.visibility import compute_ephemeris_visibility
   from across.tools.visibility.constraints import (
       SunAngleConstraint,
       MoonAngleConstraint,
       EarthLimbConstraint,
       SAAPolygonConstraint,
   )
   from shapely import Polygon

   # Step 1: Get TLE data for the spacecraft
   tle = get_tle(
       norad_id=28485,  # Swift
       epoch=datetime(2024, 1, 1),
       spacetrack_user="your_username",
       spacetrack_pwd="your_password"
   )

   # Step 2: Compute ephemeris
   ephem = compute_tle_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       step_size=60,
       tle=tle
   )

   # Step 3: Define SAA polygon
   saa = Polygon([
       (-90, -30), (-40, -30), (-30, 0), (-40, 5),
       (-90, 5), (-90, -30)
   ])

   # Step 4: Define constraints
   constraints = [
       SunAngleConstraint(min_angle=45),
       MoonAngleConstraint(min_angle=15),
       EarthLimbConstraint(min_angle=20),
       SAAPolygonConstraint(polygon=saa),
   ]

   # Step 5: Compute visibility
   visibility = compute_ephemeris_visibility(
       ra=266.417,       # Galactic Center
       dec=-29.008,
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 8),
       ephemeris=ephem,
       constraints=constraints,
       observatory_name="Swift",
       min_vis=300,      # Minimum 5 minute windows
   )

   # Step 6: Analyze results
   total_visibility = sum(w.max_visibility_duration for w in visibility.visibility_windows)
   print(f"Total visibility: {total_visibility / 3600:.1f} hours over 7 days")
   print(f"Number of windows: {len(visibility.visibility_windows)}")

   for i, window in enumerate(visibility.visibility_windows[:5], 1):
       print(f"\nWindow {i}:")
       print(f"  Start: {window.window.begin.datetime}")
       print(f"  End: {window.window.end.datetime}")
       print(f"  Duration: {window.max_visibility_duration / 60:.1f} minutes")

API Reference
-------------

See the :doc:`API Reference </autoapi/across/tools/visibility/index>` for complete 
class and function documentation.
