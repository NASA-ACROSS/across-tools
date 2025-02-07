from enum import Enum


class DepthUnit(str, Enum):
    """
    Enum to represent the astronomical depth types
    """

    AB_MAG = "ab_mag"
    VEGA_MAG = "vega_mag"
    FLUX_ERG = "flux_erg"
    FLUX_JY = "flux_jy"


class EphemerisType(str, Enum):
    """Types of ephemeris data that can be calculated."""

    GROUND_BASED = "ground_based"  # Position is fixed on Earth
    SPACE_TLE = "space_tle"  # Position calculated from a TLE
    SPACE_JPL = "space_jpl"  # Position calculated from JPL ephemeris data from Horizons
    SPACE_SPICE = "space_spice"  # Position calculated from SPICE kernel ephemeris data
