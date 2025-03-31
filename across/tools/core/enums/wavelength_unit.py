from enum import Enum


class WavelengthUnit(str, Enum):
    """
    Enum to represent the bandpass wavelength
    """

    Nanometer = "nm"
    Angstrom = "angstrom"
    Micron = "um"
    Millimeter = "mm"
