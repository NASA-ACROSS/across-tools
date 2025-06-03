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
    TEST = "Test Constraint"
