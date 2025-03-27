from enum import Enum


class WavelengthUnit(str, Enum):
    """
    Enum to represent the astronomical depth types
    """

    Nanometer = "nm"
    Angstrom = "angstrom"
    Micron = "um"
    Millimeter = "mm"
