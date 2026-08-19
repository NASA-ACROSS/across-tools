from enum import Enum


class SolarSystemObject(str, Enum):
    """Enumeration of valid solar system objects for SolarSystemConstraint calculations."""

    MERCURY = "mercury"
    VENUS = "venus"
    EARTH = "earth"
    MARS = "mars"
    JUPITER = "jupiter"
    SATURN = "saturn"
    URANUS = "uranus"
    NEPTUNE = "neptune"
