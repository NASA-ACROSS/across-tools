from enum import Enum


class TwilightType(str, Enum):
    """
    Enum to represent the astronomical twilight types
    """

    ASTRONOMICAL = "astronomical"
    NAUTICAL = "nautical"
    CIVIL = "civil"
    SUNRISE = "sunrise"
