from enum import Enum


class EphemType(str, Enum):
    """Types of ephemeris data that can be calculated."""

    ground_based = "ground_based"  # Position is fixed on Earth
    space_tle = "space_tle"  # Position calculated from a TLE
    space_jpl = "space_jpl"  # Position calculated from JPL ephemeris data from Horizons
    space_spice = "space_spice"  # Position calculated from SPICE kernel ephemeris data
