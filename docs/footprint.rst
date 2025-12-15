Footprint Analysis
========================================================================================

The ``across.tools.footprint`` module provides classes and functions for working with 
astronomical instrument fields of view (footprints). It enables projection of detector 
geometries onto the celestial sphere and spatial queries using HEALPix pixelization.

Overview
--------

The footprint module supports:

- **Footprint Definition**: Define instrument footprints as collections of polygon detectors
- **Sky Projection**: Project footprints to specific sky coordinates with arbitrary roll angles
- **HEALPix Queries**: Determine which HEALPix pixels are covered by a footprint
- **Spatial Joins**: Compute overlaps, unions, and differences between multiple footprints

This functionality is essential for:

- Planning tiled observations
- Computing sky coverage
- Cross-matching observations with catalogs
- Determining overlap between different instruments or pointings

Core Data Types
---------------

Coordinate
^^^^^^^^^^

The ``Coordinate`` class represents a point on the celestial sphere.

.. code-block:: python

   from across.tools import Coordinate

   # Create a coordinate
   coord = Coordinate(ra=180.0, dec=45.0)
   print(coord)  # Coordinate(ra=180.0, dec=45.0)

.. list-table:: Coordinate Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``ra``
     - float
     - Right ascension in degrees (-360 to 360, normalized to 0-360)
   * - ``dec``
     - float
     - Declination in degrees (-90 to 90)

Polygon
^^^^^^^

The ``Polygon`` class represents a spherical polygon defined by a list of coordinates.

.. code-block:: python

   from across.tools import Coordinate, Polygon

   # Define a square region on the sky
   square = Polygon(coordinates=[
       Coordinate(ra=0.0, dec=0.0),
       Coordinate(ra=1.0, dec=0.0),
       Coordinate(ra=1.0, dec=1.0),
       Coordinate(ra=0.0, dec=1.0),
       # First coordinate is automatically appended if not provided
   ])

   # Define a triangular region
   triangle = Polygon(coordinates=[
       Coordinate(ra=10.0, dec=10.0),
       Coordinate(ra=11.0, dec=10.0),
       Coordinate(ra=10.5, dec=11.0),
   ])

.. note::
   Polygons must have at least 3 unique coordinates. The first coordinate is 
   automatically appended to close the polygon if not already present.

.. list-table:: Polygon Requirements
   :widths: 30 70
   :header-rows: 1

   * - Requirement
     - Description
   * - Minimum vertices
     - At least 3 unique coordinates
   * - Closure
     - First and last coordinates must match (auto-appended if missing)
   * - Winding order
     - Counter-clockwise for proper HEALPix queries

Footprint Class
---------------

The ``Footprint`` class represents an instrument's imaging footprint, which can consist 
of one or more detector polygons.

Creating a Footprint
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint

   # Single detector footprint (e.g., a simple CCD)
   detector = Polygon(coordinates=[
       Coordinate(ra=-0.5, dec=-0.5),
       Coordinate(ra=0.5, dec=-0.5),
       Coordinate(ra=0.5, dec=0.5),
       Coordinate(ra=-0.5, dec=0.5),
   ])
   single_footprint = Footprint(detectors=[detector])

   # Multi-detector footprint (e.g., a mosaic camera)
   detector1 = Polygon(coordinates=[
       Coordinate(ra=-1.0, dec=-0.5),
       Coordinate(ra=0.0, dec=-0.5),
       Coordinate(ra=0.0, dec=0.5),
       Coordinate(ra=-1.0, dec=0.5),
   ])
   detector2 = Polygon(coordinates=[
       Coordinate(ra=0.0, dec=-0.5),
       Coordinate(ra=1.0, dec=-0.5),
       Coordinate(ra=1.0, dec=0.5),
       Coordinate(ra=0.0, dec=0.5),
   ])
   mosaic_footprint = Footprint(detectors=[detector1, detector2])

.. note::
   When defining footprints for projection, coordinates should be specified as 
   offsets from the boresight (pointing center) in degrees. The origin (0, 0) 
   represents the instrument boresight.

Projecting Footprints
^^^^^^^^^^^^^^^^^^^^^

The ``project()`` method projects a footprint onto the celestial sphere at a specified 
position and roll angle.

.. code-block:: python

   from across.tools import Coordinate
   from across.tools.footprint import Footprint

   # Define footprint centered at origin
   footprint = Footprint(detectors=[
       Polygon(coordinates=[
           Coordinate(ra=-0.25, dec=-0.25),
           Coordinate(ra=0.25, dec=-0.25),
           Coordinate(ra=0.25, dec=0.25),
           Coordinate(ra=-0.25, dec=0.25),
       ])
   ])

   # Project to a specific sky position
   target = Coordinate(ra=180.0, dec=45.0)
   roll_angle = 0.0  # degrees

   projected = footprint.project(coordinate=target, roll_angle=roll_angle)

   # Access projected detector coordinates
   for detector in projected.detectors:
       print("Detector vertices:")
       for coord in detector.coordinates:
           print(f"  RA={coord.ra:.4f}°, Dec={coord.dec:.4f}°")

**Roll Angle**

The roll angle rotates the footprint around the boresight axis:

.. code-block:: python

   # Project with different roll angles
   roll_0 = footprint.project(coordinate=target, roll_angle=0.0)
   roll_45 = footprint.project(coordinate=target, roll_angle=45.0)
   roll_90 = footprint.project(coordinate=target, roll_angle=90.0)

.. list-table:: project() Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``coordinate``
     - Coordinate
     - Sky position to project the footprint center to
   * - ``roll_angle``
     - float
     - Rotation angle around the boresight in degrees (-360 to 360)

HEALPix Pixel Queries
^^^^^^^^^^^^^^^^^^^^^

The ``query_pixels()`` method returns the HEALPix pixel indices covered by the footprint.

.. code-block:: python

   # Query pixels at HEALPix order 10 (default)
   pixels = projected.query_pixels()
   print(f"Footprint covers {len(pixels)} pixels at order 10")

   # Query at different orders
   pixels_low = projected.query_pixels(order=6)   # Coarse resolution
   pixels_high = projected.query_pixels(order=12) # Fine resolution

   print(f"Order 6: {len(pixels_low)} pixels")
   print(f"Order 12: {len(pixels_high)} pixels")

**HEALPix Order and Resolution**

.. list-table:: HEALPix Order Reference
   :widths: 15 20 25 40
   :header-rows: 1

   * - Order
     - NSIDE
     - Pixel Area
     - Typical Use
   * - 6
     - 64
     - ~0.84 deg²
     - Large-scale surveys
   * - 8
     - 256
     - ~0.05 deg²
     - Wide-field instruments
   * - 10
     - 1024
     - ~0.003 deg² (~12 arcmin²)
     - Standard queries (default)
   * - 12
     - 4096
     - ~0.0002 deg² (~0.7 arcmin²)
     - High-resolution instruments

.. list-table:: query_pixels() Parameters
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``order``
     - int
     - HEALPix order (1-12, default: 10)

HEALPix Spatial Joins
---------------------

The footprint module provides functions for computing spatial relationships between 
multiple footprints using HEALPix pixels.

Inner Join (Intersection)
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``inner()`` function returns pixels that are covered by **all** footprints 
(intersection/overlap region).

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint, inner

   # Helper function to create a projected footprint at a given position
   def create_footprint_at(ra: float, dec: float, roll_angle: float = 0.0) -> Footprint:
       """Create a 0.5° x 0.5° footprint projected to the given sky position."""
       detector = Polygon(coordinates=[
           Coordinate(ra=-0.25, dec=-0.25),
           Coordinate(ra=0.25, dec=-0.25),
           Coordinate(ra=0.25, dec=0.25),
           Coordinate(ra=-0.25, dec=0.25),
       ])
       base_footprint = Footprint(detectors=[detector])
       return base_footprint.project(coordinate=Coordinate(ra=ra, dec=dec), roll_angle=roll_angle)

   # Two overlapping footprints
   footprint1 = create_footprint_at(ra=180.0, dec=45.0)
   footprint2 = create_footprint_at(ra=180.1, dec=45.0)  # Slightly offset

   # Find overlapping pixels
   overlap_pixels = inner([footprint1, footprint2], order=10)
   print(f"Overlap region: {len(overlap_pixels)} pixels")

Outer Join (Non-overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``outer()`` function returns pixels that are covered by **exactly one** footprint 
(non-overlapping regions).

.. code-block:: python

   from across.tools.footprint import outer

   # Find non-overlapping pixels
   unique_pixels = outer([footprint1, footprint2], order=10)
   print(f"Non-overlapping region: {len(unique_pixels)} pixels")

Union (All Coverage)
^^^^^^^^^^^^^^^^^^^^

The ``union()`` function returns all unique pixels covered by **any** of the footprints.

.. code-block:: python

   from across.tools.footprint import union

   # Find total coverage
   all_pixels = union([footprint1, footprint2], order=10)
   print(f"Total coverage: {len(all_pixels)} pixels")

**Join Function Summary**

.. list-table:: HEALPix Join Functions
   :widths: 20 40 40
   :header-rows: 1

   * - Function
     - Returns
     - Use Case
   * - ``inner()``
     - Pixels in ALL footprints
     - Find overlap regions
   * - ``outer()``
     - Pixels in EXACTLY ONE footprint
     - Find unique coverage
   * - ``union()``
     - Pixels in ANY footprint
     - Find total sky coverage

Example: Tiled Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint, inner, outer, union

   # Define a 0.5° x 0.5° detector
   detector = Polygon(coordinates=[
       Coordinate(ra=-0.25, dec=-0.25),
       Coordinate(ra=0.25, dec=-0.25),
       Coordinate(ra=0.25, dec=0.25),
       Coordinate(ra=-0.25, dec=0.25),
   ])
   base_footprint = Footprint(detectors=[detector])

   # Create a 3x3 tiled observation pattern
   tiles = []
   for i in range(3):
       for j in range(3):
           target = Coordinate(ra=180.0 + i * 0.4, dec=45.0 + j * 0.4)
           projected = base_footprint.project(coordinate=target, roll_angle=0.0)
           tiles.append(projected)

   # Analyze coverage
   total_coverage = union(tiles, order=10)
   overlap_regions = inner(tiles, order=10)

   print(f"Total coverage: {len(total_coverage)} pixels")
   print(f"Overlap regions: {len(overlap_regions)} pixels")
   print(f"Coverage efficiency: {len(total_coverage) / (len(total_coverage) + len(overlap_regions)) * 100:.1f}%")

Working with Real Instruments
-----------------------------

Example: Simple CCD Camera
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint

   # 10 arcmin x 10 arcmin CCD
   fov = 10.0 / 60.0  # Convert arcmin to degrees
   half_fov = fov / 2

   ccd = Polygon(coordinates=[
       Coordinate(ra=-half_fov, dec=-half_fov),
       Coordinate(ra=half_fov, dec=-half_fov),
       Coordinate(ra=half_fov, dec=half_fov),
       Coordinate(ra=-half_fov, dec=half_fov),
   ])

   camera = Footprint(detectors=[ccd])

   # Point at M31
   m31 = Coordinate(ra=10.6847, dec=41.2690)
   observation = camera.project(coordinate=m31, roll_angle=0.0)

Example: Multi-CCD Mosaic Camera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint

   # 2x2 CCD mosaic with gaps
   ccd_size = 0.5  # degrees
   gap = 0.05      # degrees between CCDs

   detectors = []
   for i in range(2):
       for j in range(2):
           # Calculate CCD center position
           x_offset = (i - 0.5) * (ccd_size + gap)
           y_offset = (j - 0.5) * (ccd_size + gap)

           # Define CCD corners
           ccd = Polygon(coordinates=[
               Coordinate(ra=x_offset - ccd_size/2, dec=y_offset - ccd_size/2),
               Coordinate(ra=x_offset + ccd_size/2, dec=y_offset - ccd_size/2),
               Coordinate(ra=x_offset + ccd_size/2, dec=y_offset + ccd_size/2),
               Coordinate(ra=x_offset - ccd_size/2, dec=y_offset + ccd_size/2),
           ])
           detectors.append(ccd)

   mosaic_camera = Footprint(detectors=detectors)
   print(f"Camera has {len(mosaic_camera.detectors)} detectors")

Example: Irregular Detector Shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from across.tools import Coordinate, Polygon
   from across.tools.footprint import Footprint

   # L-shaped detector
   l_shape = Polygon(coordinates=[
       Coordinate(ra=-0.5, dec=-0.5),
       Coordinate(ra=0.0, dec=-0.5),
       Coordinate(ra=0.0, dec=0.0),
       Coordinate(ra=0.5, dec=0.0),
       Coordinate(ra=0.5, dec=0.5),
       Coordinate(ra=-0.5, dec=0.5),
   ])

   instrument = Footprint(detectors=[l_shape])

Computing Sky Coverage Statistics
---------------------------------

.. code-block:: python

   import healpy as hp
   from across.tools.footprint import Footprint, union

   def compute_coverage_stats(footprints: list[Footprint], order: int = 10) -> dict:
       """Compute sky coverage statistics for a list of footprints."""
       # Get total pixels covered
       all_pixels = union(footprints, order=order)

       # Calculate sky area
       nside = hp.order2nside(order)
       npix_total = hp.nside2npix(nside)
       pixel_area_deg2 = hp.nside2pixarea(nside, degrees=True)

       coverage_area = len(all_pixels) * pixel_area_deg2
       coverage_fraction = len(all_pixels) / npix_total

       return {
           "num_pixels": len(all_pixels),
           "area_deg2": coverage_area,
           "area_arcmin2": coverage_area * 3600,
           "sky_fraction": coverage_fraction,
           "sky_fraction_percent": coverage_fraction * 100,
       }

   # Example usage
   stats = compute_coverage_stats(tiles, order=10)
   print(f"Coverage: {stats['area_deg2']:.4f} deg² ({stats['sky_fraction_percent']:.6f}% of sky)")

Coordinate Systems and Conventions
----------------------------------

**Footprint Definition Coordinates**

When defining a footprint (before projection), coordinates represent offsets from 
the instrument boresight:

- ``ra``: Offset in the RA direction (positive = East)
- ``dec``: Offset in the Dec direction (positive = North)
- Origin (0, 0): Instrument boresight/pointing center

**Projected Coordinates**

After projection, coordinates are absolute sky positions in the ICRS frame:

- ``ra``: Right ascension (0° to 360°)
- ``dec``: Declination (-90° to 90°)

**Roll Angle Convention**

- Roll angle = 0°: Detector "up" direction aligned with celestial North
- Positive roll: Counter-clockwise rotation when viewed from behind the telescope

API Reference
-------------

See the :doc:`API Reference </autoapi/across/tools/footprint/index>` for complete 
class and function documentation.
