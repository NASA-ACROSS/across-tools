from enum import Enum


class ConstraintType(str, Enum):
    """
    Represents a constraint.
    """

    SUN = "Sun Angle"
    MOON = "Moon Angle"
    EARTH = "Earth Limb"
    WINDOW = "Window"
    UNKNOWN = "Unknown"
    SAA = "South Atlantic Anomaly"
    ALT_AZ = "Altitude/Azimuth Avoidance"
    GALACTIC_PLANE = "Galactic Plane Avoidance"
    BRIGHT_STAR = "Bright Star Avoidance"
    AIRMASS = "Airmass Limit"
    ECLIPTIC_LATITUDE = "Ecliptic Latitude"
    GALACTIC_BULGE = "Galactic Bulge Avoidance"
    SOLAR_SYSTEM = "Solar System Object Avoidance"
    DAYTIME = "Daytime Avoidance"
    TEST = "Test Constraint"
    AND = "And"
    OR = "Or"
    NOT = "Not"
    XOR = "Xor"
    POINTING = "Pointing"
