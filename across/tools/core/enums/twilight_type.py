from enum import Enum


class TwilightType(str, Enum):
    """Enumeration of twilight types for daytime constraints."""

    ASTRONOMICAL = "astronomical"  # Sun 18° below horizon
    NAUTICAL = "nautical"  # Sun 12° below horizon
    CIVIL = "civil"  # Sun 6° below horizon
    SUNSET = "sunset"  # Sun at horizon
