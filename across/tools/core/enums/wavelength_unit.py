from enum import Enum


class WavelengthUnit(str, Enum):
    """
    Enum to represent the bandpass wavelength
    """

    NANOMETER = "nm"
    ANGSTROM = "angstrom"
    MICRON = "um"
    MILLIMETER = "mm"
