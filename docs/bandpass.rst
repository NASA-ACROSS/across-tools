Bandpass & Spectral Types
========================================================================================

The ``across.tools`` library provides classes for defining spectral bandpasses across 
different domains: wavelength, energy, and frequency. These are useful for specifying 
the observational bands of instruments and can be converted between different 
representations.

Overview
--------

Bandpasses define the spectral range over which an instrument is sensitive. The library 
supports three types of bandpass definitions:

- **WavelengthBandpass**: Define bands in wavelength units (nm, Å, μm, mm)
- **EnergyBandpass**: Define bands in energy units (eV, keV, MeV, GeV, TeV)
- **FrequencyBandpass**: Define bands in frequency units (Hz, kHz, MHz, GHz, THz)

All bandpass types can be converted to wavelength representation for consistent internal 
processing.

Importing Bandpass Classes
--------------------------

.. code-block:: python

   from across.tools import (
       WavelengthBandpass,
       EnergyBandpass,
       FrequencyBandpass,
   )
   
   # Also import the unit enums
   from across.tools.core.enums import (
       WavelengthUnit,
       EnergyUnit,
       FrequencyUnit,
   )

WavelengthBandpass
------------------

Use ``WavelengthBandpass`` for optical, infrared, and UV observations.

Basic Usage
^^^^^^^^^^^

You can define a wavelength bandpass using either min/max values or 
central_wavelength/bandwidth:

.. code-block:: python

   from across.tools import WavelengthBandpass
   from across.tools.core.enums import WavelengthUnit

   # Using min/max range
   v_band = WavelengthBandpass(
       filter_name="V-band",
       min=500,
       max=600,
       unit=WavelengthUnit.NANOMETER
   )

   print(f"Filter: {v_band.filter_name}")
   print(f"Central wavelength: {v_band.central_wavelength} Å")
   print(f"Bandwidth: {v_band.bandwidth} Å")
   print(f"Range: {v_band.min} - {v_band.max} Å")

   # Using central wavelength and bandwidth directly
   r_band = WavelengthBandpass(
       filter_name="R-band",
       central_wavelength=658,
       bandwidth=138,
       unit=WavelengthUnit.NANOMETER
   )

.. note::

   All wavelength values are automatically converted to Angstroms (Å) internally 
   for consistency.

Attributes
^^^^^^^^^^

.. list-table:: WavelengthBandpass Attributes
   :widths: 25 20 55
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``filter_name``
     - str | None
     - Optional name for the filter (e.g., "V-band", "SDSS-g")
   * - ``min``
     - float | None
     - Minimum wavelength of the bandpass range
   * - ``max``
     - float | None
     - Maximum wavelength of the bandpass range
   * - ``central_wavelength``
     - float | None
     - Central wavelength of the filter
   * - ``peak_wavelength``
     - float | None
     - Peak transmission wavelength (if different from central)
   * - ``bandwidth``
     - float | None
     - Half-width of the bandpass
   * - ``unit``
     - WavelengthUnit
     - Unit of measurement (converted to Angstrom internally)

Wavelength Units
^^^^^^^^^^^^^^^^

.. list-table:: Available Wavelength Units
   :widths: 30 30 40
   :header-rows: 1

   * - Unit
     - Value
     - Common Use
   * - ``WavelengthUnit.ANGSTROM``
     - ``"angstrom"``
     - X-ray, UV, optical spectroscopy
   * - ``WavelengthUnit.NANOMETER``
     - ``"nm"``
     - Optical, UV observations
   * - ``WavelengthUnit.MICRON``
     - ``"um"``
     - Infrared observations
   * - ``WavelengthUnit.MILLIMETER``
     - ``"mm"``
     - Submillimeter observations

EnergyBandpass
--------------

Use ``EnergyBandpass`` for X-ray and gamma-ray observations.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from across.tools import EnergyBandpass
   from across.tools.core.enums import EnergyUnit

   # Define an X-ray bandpass (0.3-10 keV)
   soft_xray = EnergyBandpass(
       filter_name="Soft X-ray",
       min=0.3,
       max=10.0,
       unit=EnergyUnit.keV
   )

   print(f"Filter: {soft_xray.filter_name}")
   print(f"Energy range: {soft_xray.min} - {soft_xray.max} {soft_xray.unit.value}")

   # High-energy gamma-ray band
   gamma_ray = EnergyBandpass(
       filter_name="HE Gamma",
       min=100,
       max=300,
       unit=EnergyUnit.GeV
   )

Attributes
^^^^^^^^^^

.. list-table:: EnergyBandpass Attributes
   :widths: 25 20 55
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``filter_name``
     - str | None
     - Optional name for the filter
   * - ``min``
     - float
     - Minimum energy of the bandpass range
   * - ``max``
     - float
     - Maximum energy of the bandpass range
   * - ``unit``
     - EnergyUnit
     - Unit of measurement for energy

Energy Units
^^^^^^^^^^^^

.. list-table:: Available Energy Units
   :widths: 25 25 50
   :header-rows: 1

   * - Unit
     - Value
     - Common Use
   * - ``EnergyUnit.eV``
     - ``"eV"``
     - UV, soft X-ray boundary
   * - ``EnergyUnit.keV``
     - ``"keV"``
     - X-ray observations (Swift, Chandra, XMM)
   * - ``EnergyUnit.MeV``
     - ``"MeV"``
     - Soft gamma-rays
   * - ``EnergyUnit.GeV``
     - ``"GeV"``
     - High-energy gamma-rays (Fermi-LAT)
   * - ``EnergyUnit.TeV``
     - ``"TeV"``
     - Very high-energy gamma-rays (Cherenkov telescopes)

FrequencyBandpass
-----------------

Use ``FrequencyBandpass`` for radio observations.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from across.tools import FrequencyBandpass
   from across.tools.core.enums import FrequencyUnit

   # Define a radio bandpass
   radio_band = FrequencyBandpass(
       filter_name="L-band",
       min=1.0,
       max=2.0,
       unit=FrequencyUnit.GHz
   )

   print(f"Filter: {radio_band.filter_name}")
   print(f"Frequency range: {radio_band.min} - {radio_band.max} {radio_band.unit.value}")

   # Millimeter-wave observation
   mm_wave = FrequencyBandpass(
       filter_name="ALMA Band 6",
       min=211,
       max=275,
       unit=FrequencyUnit.GHz
   )

Attributes
^^^^^^^^^^

.. list-table:: FrequencyBandpass Attributes
   :widths: 25 20 55
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``filter_name``
     - str | None
     - Optional name for the filter
   * - ``min``
     - float
     - Minimum frequency of the bandpass range
   * - ``max``
     - float
     - Maximum frequency of the bandpass range
   * - ``unit``
     - FrequencyUnit
     - Unit of measurement for frequency

Frequency Units
^^^^^^^^^^^^^^^

.. list-table:: Available Frequency Units
   :widths: 25 25 50
   :header-rows: 1

   * - Unit
     - Value
     - Common Use
   * - ``FrequencyUnit.Hz``
     - ``"Hz"``
     - Low frequency radio
   * - ``FrequencyUnit.kHz``
     - ``"kHz"``
     - Long-wave radio
   * - ``FrequencyUnit.MHz``
     - ``"MHz"``
     - FM radio, low-frequency astronomy
   * - ``FrequencyUnit.GHz``
     - ``"GHz"``
     - Microwave, radio astronomy
   * - ``FrequencyUnit.THz``
     - ``"THz"``
     - Submillimeter astronomy

Converting Between Domains
--------------------------

The ``convert_to_wave()`` function converts energy or frequency bandpasses to 
wavelength representation:

.. code-block:: python

   from across.tools import EnergyBandpass, FrequencyBandpass
   from across.tools.core.enums import EnergyUnit, FrequencyUnit
   from across.tools.core.schemas.bandpass import convert_to_wave

   # Convert X-ray energy band to wavelength
   xray_band = EnergyBandpass(
       filter_name="Swift XRT",
       min=0.3,
       max=10.0,
       unit=EnergyUnit.keV
   )
   
   xray_wavelength = convert_to_wave(xray_band)
   print(f"X-ray band in wavelength: {xray_wavelength.min:.2f} - {xray_wavelength.max:.2f} Å")

   # Convert radio frequency band to wavelength
   radio_band = FrequencyBandpass(
       filter_name="VLA C-band",
       min=4.0,
       max=8.0,
       unit=FrequencyUnit.GHz
   )
   
   radio_wavelength = convert_to_wave(radio_band)
   print(f"Radio band in wavelength: {radio_wavelength.min:.2e} - {radio_wavelength.max:.2e} Å")

.. important::

   When converting from energy or frequency to wavelength, the min/max values are 
   inverted. High energy corresponds to short wavelength, and high frequency 
   corresponds to short wavelength. The ``convert_to_wave()`` function handles 
   this automatically.

Spectral Domain Relationships
-----------------------------

The electromagnetic spectrum can be described in terms of wavelength (λ), 
frequency (ν), or energy (E). These are related by:

.. math::

   E = h \nu = \frac{hc}{\lambda}

Where:

- :math:`h` is Planck's constant (:math:`6.626 \times 10^{-34}` J·s)
- :math:`c` is the speed of light (:math:`3 \times 10^8` m/s)
- :math:`\lambda` is wavelength
- :math:`\nu` is frequency

Common Bandpass Examples
------------------------

Here are some example bandpasses for common astronomical observations:

.. code-block:: python

   from across.tools import WavelengthBandpass, EnergyBandpass, FrequencyBandpass
   from across.tools.core.enums import WavelengthUnit, EnergyUnit, FrequencyUnit

   # Optical bands
   johnson_v = WavelengthBandpass(
       filter_name="Johnson V",
       central_wavelength=551,
       bandwidth=88,
       unit=WavelengthUnit.NANOMETER
   )

   sdss_g = WavelengthBandpass(
       filter_name="SDSS g",
       min=400,
       max=550,
       unit=WavelengthUnit.NANOMETER
   )

   # X-ray bands
   swift_xrt = EnergyBandpass(
       filter_name="Swift XRT",
       min=0.3,
       max=10.0,
       unit=EnergyUnit.keV
   )

   chandra_acis = EnergyBandpass(
       filter_name="Chandra ACIS",
       min=0.5,
       max=8.0,
       unit=EnergyUnit.keV
   )

   # Gamma-ray bands
   fermi_lat = EnergyBandpass(
       filter_name="Fermi LAT",
       min=100,
       max=300000,
       unit=EnergyUnit.MeV
   )

   # Radio bands
   vla_l_band = FrequencyBandpass(
       filter_name="VLA L-band",
       min=1.0,
       max=2.0,
       unit=FrequencyUnit.GHz
   )

Error Handling
--------------

The bandpass classes validate input values using Pydantic validation. When validation 
fails, a ``pydantic.ValidationError`` is raised containing details about the error:

.. code-block:: python

   from across.tools import WavelengthBandpass, EnergyBandpass
   from across.tools.core.enums import WavelengthUnit, EnergyUnit
   from pydantic import ValidationError

   # Error: max less than min
   try:
       invalid = WavelengthBandpass(
           min=600,
           max=400,  # Error: max < min
           unit=WavelengthUnit.NANOMETER
       )
   except ValidationError as e:
       print(f"Error: {e}")

   # Error: negative values
   try:
       invalid = EnergyBandpass(
           min=-1.0,  # Error: negative value
           max=10.0,
           unit=EnergyUnit.keV
       )
   except ValidationError as e:
       print(f"Error: {e}")

   # Error: only one of min/max provided
   try:
       invalid = WavelengthBandpass(
           min=500,
           # max not provided
           unit=WavelengthUnit.NANOMETER
       )
   except ValidationError as e:
       print(f"Error: {e}")

**Common Validation Errors**:

- Max wavelength/energy/frequency less than min
- Negative values for wavelength, energy, or frequency
- Only one of min/max provided (both are required)
- Missing central_wavelength or bandwidth for WavelengthBandpass

API Reference
-------------

See the :doc:`API Reference </autoapi/across/tools/core/schemas/bandpass/index>` for 
complete class and function documentation.
