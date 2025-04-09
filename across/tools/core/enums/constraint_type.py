from enum import Enum


class ConstraintType(str, Enum):
    """
    Represents a constraint.
    """

    SUN = "Sun"
    MOON = "Moon"
    EARTH = "Earth"
    UNKNOWN = "Unknown"
    WINDOW = "Window"
    VISIBILITY = "Visibility"
    DAY = "Day"
    MOON_PHASE = "Moon Phase"
    ORBIT_DAY = "Orbit Day"
    ORBIT_POLE = "Orbit Pole"
    ORBIT_RAM = "Orbit Ram"
    ORBIT_SAA = "SAA"
    AIR_MASS = "Air Mass"
    ALT_AZ = "Alt/Az"
    FOV = "FOV"
