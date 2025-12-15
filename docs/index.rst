
.. across-tools documentation main file.

across-tools Documentation
========================================================================================

**across-tools** is a Python library for astronomical observatory visibility calculations, 
ephemeris generation, and instrument footprint analysis. It provides tools for computing 
when celestial targets are observable from space-based and ground-based observatories, 
accounting for various observational constraints.

**across-tools** is developed and maintained by NASA's Astrophysics
Cross Observatory Science Support (ACROSS) Team, and is part of the larger
ACROSS software ecosystem, which includes APIs and API clients for accessing
information about astronomical observatories, instruments, and observation
planning. **across-tools** is designed to be a shared module that focuses on
core computational functionality, while higher-level interfaces and user-facing
tools are provided by other ACROSS packages.

Features
--------

- **Ephemeris Calculation**: Generate ephemerides for spacecraft and ground-based observatories using TLE, JPL Horizons, SPICE kernels, or ground station coordinates.

- **Visibility Computation**: Determine target visibility windows with support for Sun angle, Moon angle, Earth limb, SAA avoidance, and altitude/azimuth constraints.

- **Footprint Analysis**: Work with instrument field-of-view footprints, project them onto the celestial sphere, and perform HEALPix-based spatial queries.

Installation
------------

Install from PyPI:

.. code-block:: console

   pip install across-tools

Or install from source:

.. code-block:: console

   git clone https://github.com/NASA-ACROSS/across-tools.git
   cd across-tools
   pip install -e .

Quick Start
-----------

Computing Ephemeris from TLE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.ephemeris import compute_tle_ephemeris

   # Compute ephemeris for a spacecraft using TLE
   ephem = compute_tle_ephemeris(
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 2),
       tle_line1="1 00000U 00000A   24001.00000000  .00000000  00000-0  00000-0 0    09",
       tle_line2="2 00000   0.0000   0.0000 0000000   0.0000   0.0000 15.00000000    03"
   )

Computing Target Visibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from datetime import datetime
   from across.tools.visibility import compute_ephemeris_visibility
   from across.tools.visibility.constraints import SunAngleConstraint

   # Calculate visibility of a target with Sun angle constraint
   visibility = compute_ephemeris_visibility(
       ra=180.0,  # Right ascension in degrees
       dec=45.0,  # Declination in degrees
       begin=datetime(2024, 1, 1),
       end=datetime(2024, 1, 7),
       constraints=[SunAngleConstraint(min=45, max=180)],
       ephemeris=ephem
   )

   # Access visibility windows
   for window in visibility.visibility_windows:
       print(f"Visible from {window.begin} to {window.end}")

Development
-----------

Setting Up Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to help development or contribute to the project, follow these steps to set up
a development environment using a python virtual environment:

.. code-block:: console

   # Clone the repo
   git clone git@github.com:NASA-ACROSS/across-tools.git
   cd across-tools

   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate 

   # Install development dependencies
   pip install -e '.[dev]'

   # Install pre-commit hooks
   pre-commit install

Note: please be sure in to run the last command to install the pre-commit hooks, which will
help ensure code quality and consistency. 

.. toctree::
   :hidden:

   Home page <self>
   Ephemeris Generation <ephemeris>
   TLE Data <tle>
   Visibility Calculation <visibility>
   Footprint Analysis <footprint>
   Bandpass & Spectral Types <bandpass>
   API Reference <autoapi/index>
   Notebooks <notebooks>
