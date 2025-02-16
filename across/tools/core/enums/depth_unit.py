from enum import Enum


class DepthUnit(str, Enum):
    """
    Enum to represent the astronomical depth types
    """

    AB_MAG = "ab_mag"
    VEGA_MAG = "vega_mag"
    FLUX_ERG = "flux_erg"
    FLUX_JY = "flux_jy"


class TwilightType(str, Enum):
    """
    Enum to represent the astronomical twilight types
    """

    ASTRONOMICAL = "astronomical"
    NAUTICAL = "nautical"
    CIVIL = "civil"
    SUNRISE = "sunrise"
